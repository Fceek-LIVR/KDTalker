import random
import time
import cv2
import numpy as np
from rich.progress import track
import torch
from PIL import Image
import argparse
import sys

from tqdm import tqdm

import inference
from src.utils.camera import get_rotation_matrix
import sounddevice as sd
from queue import Queue
import threading
import queue

class RTInferencer(inference.Inferencer):

    image_cache = dict()

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def pre_process_image(self, image_path):
        """
        This will pre-process the given image at avatar-creation time, save some time
        """
        image = np.array(Image.open(image_path).convert('RGB'))
        cropped_image, crop, quad = self.croper.crop([image], still=False, xsize=512)
        input_image = cv2.resize(cropped_image[0], (256, 256))

        I_s = torch.FloatTensor(input_image.transpose((2, 0, 1))).unsqueeze(0).cuda() / 255

        x_s_info = self.live_portrait_pipeline.live_portrait_wrapper.get_kp_info(I_s)
        x_c_s = x_s_info['kp'].reshape(1, 21, -1)
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        f_s = self.live_portrait_pipeline.live_portrait_wrapper.extract_feature_3d(I_s)
        x_s = self.live_portrait_pipeline.live_portrait_wrapper.transform_keypoint(x_s_info)

        kp_info = {}
        for k in x_s_info.keys():
            kp_info[k] = x_s_info[k].cpu().numpy()

        kp_info = self._norm(kp_info)

        ori_kp = torch.cat([torch.zeros([1, 7]), torch.Tensor(kp_info['kp'])], -1).cuda()

        input_x = np.concatenate([kp_info[k] for k in ['scale', 'yaw', 'pitch', 'roll', 't', 'exp']], 1)
        input_x = np.expand_dims(input_x, -1)
        input_x = np.expand_dims(input_x, 0)
        input_x = np.concatenate([input_x, input_x, input_x], -1)

        self.image_cache[image_path] = {
            'outputs': [input_x],
            'ori_kp': ori_kp,
            'x_c_s': x_c_s,
            'R_s': R_s,
            'f_s': f_s,
            'x_s': x_s,
            'x_s_info': x_s_info,
        }
        return self.image_cache[image_path]

    def process_audio_chunk(self, wav):
        fps = 25
        sr = 16000
        syncnet_mel_step_size = 16

        # Calculate number of frames for this chunk

        bit_per_frames = sr / fps
        num_frames = int(len(wav) / bit_per_frames)
        audio_length = int(num_frames * bit_per_frames)
        wav = wav[:audio_length] if len(wav) > audio_length else np.pad(wav, [0, audio_length - len(wav)], mode='constant', constant_values=0)

        orig_mel = inference.audio.melspectrogram(wav).T
        spec = orig_mel.copy()
        indiv_mels = []

        for i in range(num_frames):
            start_frame_num = i - 2
            start_idx = int(80. * (start_frame_num / float(fps)))
            end_idx = start_idx + syncnet_mel_step_size
            seq = list(range(start_idx, end_idx))
            seq = [min(max(item, 0), orig_mel.shape[0] - 1) for item in seq]
            m = spec[seq, :]
            indiv_mels.append(m.T)
        indiv_mels = np.asarray(indiv_mels)  # T 80 16
        indiv_mels = torch.FloatTensor(indiv_mels).cuda().unsqueeze(0).unsqueeze(2)
        with torch.no_grad():
            hidden = self.wav2lip_model(indiv_mels)
        audio_feat = hidden[0].cpu().detach().numpy()
        return audio_feat

    def smooth(self, sequence, n_dim_state=1):
        kf = inference.KalmanFilter(initial_state_mean=sequence[0],
                        transition_covariance=0.05 * np.eye(n_dim_state),
                        observation_covariance=0.001 * np.eye(n_dim_state))
        state_means, _ = kf.smooth(sequence)
        return state_means

    def start_persistent_window(self, window_name="RTA Monitor", fps=25):
        self.frame_queue = queue.Queue()
        self.display_queue = queue.Queue()  # Queue for main thread to process
        self.window_name = window_name
        self.fps = fps
        self._stop_display = False

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 512, 512)
        cv2.moveWindow(window_name, 100, 100)  # Position window on screen

        def display_loop():
            """Background thread that processes frames and sends them to main thread"""
            while not self._stop_display:
                try:
                    # Wait for new frames, timeout to allow checking for stop signal
                    I_p_lst = self.frame_queue.get(timeout=0.1)
                    
                    processed_frames = []
                    for frame in I_p_lst:
                        # Ensure frame is in the correct format for display
                        if isinstance(frame, torch.Tensor):
                            frame = frame.cpu().numpy()
                        
                        # Ensure frame is in uint8 format with correct range
                        if frame.dtype != np.uint8:
                            if frame.max() <= 1.0:
                                frame = (frame * 255).astype(np.uint8)
                            else:
                                frame = frame.astype(np.uint8)
                        
                        # Ensure frame has correct shape and channels
                        if len(frame.shape) == 3 and frame.shape[2] == 3:
                            # Convert RGB to BGR for OpenCV
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        else:
                            frame_bgr = frame
                        
                        frame_bgr = cv2.resize(frame_bgr, (512, 512))
                        processed_frames.append(frame_bgr)
                    
                    # Send processed frames to main thread for display
                    self.display_queue.put(processed_frames)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Error in display loop: {e}")
                    import traceback
                    traceback.print_exc()

        self.display_thread = threading.Thread(target=display_loop, daemon=True)
        self.display_thread.start()

    def stream_new_animation(self, I_p_lst):
        """Call this every time you have a new animation to display."""
        if hasattr(self, 'frame_queue'):
            self.frame_queue.put(I_p_lst)

    def process_display_queue(self):
        """Call this from main thread to process and display frames"""
        try:
            while not self.display_queue.empty():
                processed_frames = self.display_queue.get_nowait()
                for frame in processed_frames:
                    cv2.imshow(self.window_name, frame)
                    key = cv2.waitKey(int(1000 / self.fps))
                    if key == 27:  # ESC to exit
                        self._stop_display = True
                        return False
            return True
        except queue.Empty:
            return True

    def start_concurrent_processing(self, cache, window_name="RTA Monitor", fps=25):
        """Start concurrent processing system with separate threads for recording, processing, and display"""
        self.latest_audio_chunk = None
        self.processing_queue = Queue()
        self.display_queue = queue.Queue()
        self.window_name = window_name
        self.fps = fps
        self._stop_processing = False
        
        # Cache data
        self.cache = cache
        self.outputs = cache['outputs'].copy()
        self.ori_kp = cache['ori_kp']
        self.x_c_s = cache['x_c_s']
        self.R_s = cache['R_s']
        self.f_s = cache['f_s']
        self.x_s = cache['x_s']
        self.x_s_info = cache['x_s_info']

        # Setup display window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 512, 512)
        cv2.moveWindow(window_name, 100, 100)

        def processing_loop():
            """Background thread for audio processing and animation generation"""
            start_time = None
            def report_time_usage(flag: str):
                nonlocal start_time
                if start_time is None:
                    start_time = time.perf_counter_ns()
                    return
                current_time = time.perf_counter_ns()
                print(f"{flag} time: {(current_time - start_time) / 1_000_000} ms")
                start_time = current_time
            while not self._stop_processing:
                try:
                    # Get the most recent audio chunk
                    if self.latest_audio_chunk is None:
                        time.sleep(0.01)
                        continue
                    audio_chunk = self.latest_audio_chunk
                    # audio_chunk = RTInferencer.generate_random_audio_chunk(blocksize=6400, channels=1, dtype='int16')
                    freq = np.random.randint(100, 1000)
                    audio_chunk = RTInferencer.generate_sine_wave(freq=freq, duration=0.4, sr=16000)
                    self.latest_audio_chunk = None
                    wav = audio_chunk.flatten()

                    report_time_usage("Audio callback")
                    # Process audio
                    audio_feat = self.process_audio_chunk(wav)
                    report_time_usage("Audio preprocess")

                    print("audio_feat shape:", audio_feat.shape)
                    # Process keypoints
                    sample_frame = 10
                    padding_size = (sample_frame - audio_feat.shape[0] % sample_frame) % sample_frame
                    if padding_size > 0:
                        audio_feat = np.concatenate((audio_feat, audio_feat[:padding_size, :]), axis=0)

                    print("audio_feat shape after padding:", audio_feat.shape)

                    temp_outputs = self.outputs.copy()
                    # Generate keypoints
                    for i in range(0, audio_feat.shape[0] - 1, sample_frame):
                        input_mel = torch.Tensor(audio_feat[i: i + sample_frame]).unsqueeze(0).cuda()
                        kp0 = torch.Tensor(temp_outputs[-1])[:, -1].cuda()
                        pred_kp = self.point_diffusion.forward_sample(
                            70, ref_kps=kp0, ori_kps=self.ori_kp, aud_feat=input_mel, 
                            scheduler='ddim', num_inference_steps=50, disable_tqdm=True
                        )
                        temp_outputs.append(pred_kp.cpu().numpy())
                    report_time_usage("Point diffusion")

                    # Process outputs
                    outputs = np.mean(np.concatenate(temp_outputs, 1)[0, 1:audio_feat.shape[0] + 1], -1)
                    output_dict = self.output_to_dict(outputs)
                    output_dict = self._denorm(output_dict)

                    # Generate animation frames
                    I_p_lst = self._generate_animation_frames(start_time, output_dict)
                    report_time_usage("Animation")
                    # Send to display thread
                    self.display_queue.put(I_p_lst)
                    print(f"Sent {len(I_p_lst)} frames to display thread")
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Error in processing loop: {e}")
                    import traceback
                    traceback.print_exc()

        # Start only the processing thread
        self.processing_thread = threading.Thread(target=processing_loop, daemon=True)
        self.processing_thread.start()

    def _generate_animation_frames(self, start_time, output_dict):
        """Generate animation frames from output dictionary"""
        num_frame = output_dict['yaw'].shape[0]
        x_d_info = {}
        for key in output_dict:
            x_d_info[key] = torch.tensor(output_dict[key]).cuda()

        # Smooth the data
        yaw_data = x_d_info['yaw'].cpu().numpy()
        pitch_data = x_d_info['pitch'].cpu().numpy()
        roll_data = x_d_info['roll'].cpu().numpy()
        t_data = x_d_info['t'].cpu().numpy()
        exp_data = x_d_info['exp'].cpu().numpy()
        
        smoothed_pitch = self.smooth(pitch_data, n_dim_state=1)
        smoothed_yaw = self.smooth(yaw_data, n_dim_state=1)
        smoothed_roll = self.smooth(roll_data, n_dim_state=1)
        smoothed_t = self.smooth(t_data, n_dim_state=3)
        smoothed_exp = self.smooth(exp_data, n_dim_state=63)

        x_d_info['pitch'] = torch.Tensor(smoothed_pitch).cuda()
        x_d_info['yaw'] = torch.Tensor(smoothed_yaw).cuda()
        x_d_info['roll'] = torch.Tensor(smoothed_roll).cuda()
        x_d_info['t'] = torch.Tensor(smoothed_t).cuda()
        x_d_info['exp'] = torch.Tensor(smoothed_exp).cuda()

        print(f"Smoothing time: {(time.perf_counter_ns() - start_time) / 1_000_000} ms")

        # Generate frames
        I_p_lst = []
        R_d_0, x_d_0_info = None, None

        for i in range(num_frame):
            x_d_i_info = {
                'scale': x_d_info['scale'][i].cpu().numpy().astype(np.float32),
                'R_d': get_rotation_matrix(x_d_info['pitch'][i], x_d_info['yaw'][i], x_d_info['roll'][i]).cpu().numpy().astype(np.float32),
                'exp': x_d_info['exp'][i].reshape(1, 21, -1).cpu().numpy().astype(np.float32),
                't': x_d_info['t'][i].cpu().numpy().astype(np.float32),
            }
            
            for key in x_d_i_info:
                x_d_i_info[key] = torch.tensor(x_d_i_info[key]).cuda()
            R_d_i = x_d_i_info['R_d']

            if i == 0:
                R_d_0 = R_d_i
                x_d_0_info = x_d_i_info

            if self.inf_cfg.flag_relative_motion:
                R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ self.R_s
                delta_new = self.x_s_info['exp'].reshape(1, 21, -1) + (x_d_i_info['exp'] - x_d_0_info['exp'])
                scale_new = self.x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
                t_new = self.x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
            else:
                R_new = R_d_i
                delta_new = x_d_i_info['exp']
                scale_new = self.x_s_info['scale']
                t_new = x_d_i_info['t']

            t_new[..., 2].fill_(0)
            x_d_i_new = scale_new * (self.x_c_s @ R_new + delta_new) + t_new

            out = self.live_portrait_pipeline.live_portrait_wrapper.warp_decode(self.f_s, self.x_s, x_d_i_new)
            I_p_i = self.live_portrait_pipeline.live_portrait_wrapper.parse_output(out['out'])[0]
            I_p_lst.append(I_p_i)

        return I_p_lst

    def stop_concurrent_processing(self):
        """Stop all concurrent processing threads"""
        self._stop_processing = True
        cv2.destroyAllWindows()

    @staticmethod
    def generate_random_audio_chunk(blocksize=3200, channels=1, dtype='int16'):
        """
        Generate a random audio chunk for testing purposes.
        Args:
            blocksize (int): Number of samples per chunk (default 3200 for 0.2s at 16kHz)
            channels (int): Number of audio channels (default 1)
            dtype (str): Data type of the audio (default 'int16')
        Returns:
            np.ndarray: Random audio chunk of shape (blocksize, channels)
        Usage:
            fake_chunk = RTInferencer.generate_random_audio_chunk()
        """
        # Generate random int16 audio data in the range of typical PCM audio
        return np.random.randint(-32768, 32767, size=(blocksize, channels), dtype=np.int16)

    @staticmethod
    def generate_sine_wave(freq=440, duration=0.2, sr=16000):
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        # Convert to int16 for your pipeline
        return (audio * 32767).astype(np.int16)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-au", type=str, default="example/audio_driven/WDA_BenCardin1_000.wav", help="the audio")
    parser.add_argument("-img", type=str, default="example/source_image/WDA_BenCardin1_000.png", help="the image")
    parser.add_argument("--concurrent", action="store_true", help="Use concurrent processing mode")

    args = parser.parse_args()
    
    def report_time_usage(flag: str):
        global start_time
        if start_time is None:
            return
        current_time = time.perf_counter_ns()
        print(f"{flag} time: {(current_time - start_time) / 1_000_000} ms")
        start_time = current_time

    # Initialize inferencer
    rt_inferencer = RTInferencer()
    if (args.img in rt_inferencer.image_cache):
        cache = rt_inferencer.image_cache[args.img]
    else:
        cache = rt_inferencer.pre_process_image(args.img)

    if args.concurrent:
        rt_inferencer.start_concurrent_processing(cache)
        def audio_callback(indata, frames, time, status):
            rt_inferencer.latest_audio_chunk = indata.copy()
        with sd.InputStream(samplerate=16000, blocksize=6400, channels=1, dtype='int16', callback=audio_callback):
            try:
                while not rt_inferencer._stop_processing:
                    try:
                        I_p_lst = rt_inferencer.display_queue.get(timeout=0.1)
                        print(f"Received {len(I_p_lst)} frames from processing thread")
                        for frame in I_p_lst:
                            if isinstance(frame, torch.Tensor):
                                frame = frame.cpu().numpy()
                            if frame.dtype != np.uint8:
                                if frame.max() <= 1.0:
                                    frame = (frame * 255).astype(np.uint8)
                                else:
                                    frame = frame.astype(np.uint8)
                            if len(frame.shape) == 3 and frame.shape[2] == 3:
                                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            else:
                                frame_bgr = frame
                            frame_bgr = cv2.resize(frame_bgr, (512, 512))
                            cv2.imshow(rt_inferencer.window_name, frame_bgr)
                            key = cv2.waitKey(int(1000 / rt_inferencer.fps))
                            if key == 27:  # ESC to exit
                                rt_inferencer._stop_processing = True
                                break
                    except queue.Empty:
                        continue
            except KeyboardInterrupt:
                print("Stopping concurrent processing...")
            finally:
                cv2.destroyAllWindows()
                rt_inferencer.stop_concurrent_processing()
    else:
        # Original sequential mode
        rt_inferencer.start_persistent_window()
        
        outputs = cache['outputs']
        ori_kp = cache['ori_kp']
        x_c_s = cache['x_c_s']
        R_s = cache['R_s']
        f_s = cache['f_s']
        x_s = cache['x_s']
        x_s_info = cache['x_s_info']
        audio_queue = Queue()
                
        def audio_callback(indata, frames, time, status):
            audio_queue.put(indata.copy())

        with sd.InputStream(samplerate=16000, blocksize=3200, channels=1, dtype='int16', callback=audio_callback):
            chunk_counter = 0
            while True:
                try:
                    if not audio_queue.empty():
                        
                        outputs = cache['outputs']
                        audio_chunk = audio_queue.get()
                        chunk_counter += 1
                        wav = audio_chunk.flatten()

                        start_time = time.perf_counter_ns()
                        report_time_usage("Audio callback")
                        audio_feat = rt_inferencer.process_audio_chunk(wav)
                        report_time_usage("Audio preprocess")
                        sample_frame = 64
                        padding_size = (sample_frame - audio_feat.shape[0] % sample_frame) % sample_frame

                        if padding_size > 0:
                            audio_feat = np.concatenate((audio_feat, audio_feat[:padding_size, :]), axis=0)
                        else:
                            audio_feat = audio_feat

                        for i in range(0, audio_feat.shape[0] - 1, sample_frame):
                            input_mel = torch.Tensor(audio_feat[i: i + sample_frame]).unsqueeze(0).cuda()
                            kp0 = torch.Tensor(outputs[-1])[:, -1].cuda()
                            pred_kp= rt_inferencer.point_diffusion.forward_sample(
                                70,
                                ref_kps=kp0,
                                ori_kps=ori_kp,
                                aud_feat=input_mel, 
                                scheduler='ddim',
                                num_inference_steps=50,
                                disable_tqdm=True
                            )
                            outputs.append(pred_kp.cpu().numpy())
                        report_time_usage("Point diffusion")
                        outputs = np.mean(np.concatenate(outputs, 1)[0, 1:audio_feat.shape[0] + 1], -1)
                        output_dict = rt_inferencer.output_to_dict(outputs)
                        output_dict = rt_inferencer._denorm(output_dict)

                        num_frame = output_dict['yaw'].shape[0]
                        x_d_info = {}
                        for key in output_dict:
                            x_d_info[key] = torch.tensor(output_dict[key]).cuda()

                        yaw_data = x_d_info['yaw'].cpu().numpy()
                        pitch_data = x_d_info['pitch'].cpu().numpy()
                        roll_data = x_d_info['roll'].cpu().numpy()
                        t_data = x_d_info['t'].cpu().numpy()
                        exp_data = x_d_info['exp'].cpu().numpy()
                        yaw_data = rt_inferencer.smooth(yaw_data)

                        smoothed_pitch = rt_inferencer.smooth(pitch_data, n_dim_state=1)
                        smoothed_yaw = rt_inferencer.smooth(yaw_data, n_dim_state=1)
                        smoothed_roll = rt_inferencer.smooth(roll_data, n_dim_state=1)
                        smoothed_t = rt_inferencer.smooth(t_data, n_dim_state=3)
                        smoothed_exp = rt_inferencer.smooth(exp_data, n_dim_state=63)

                        x_d_info['pitch'] = torch.Tensor(smoothed_pitch).cuda()
                        x_d_info['yaw'] = torch.Tensor(smoothed_yaw).cuda()
                        x_d_info['roll'] = torch.Tensor(smoothed_roll).cuda()
                        x_d_info['t'] = torch.Tensor(smoothed_t).cuda()
                        x_d_info['exp'] = torch.Tensor(smoothed_exp).cuda()

                        report_time_usage("Smoothing")

                        template_dct = {'motion': [], 'c_d_eyes_lst': [], 'c_d_lip_lst': []}
                        for i in range(num_frame):
                            x_d_i_info = x_d_info
                            R_d_i = get_rotation_matrix(x_d_i_info['pitch'][i], x_d_i_info['yaw'][i], x_d_i_info['roll'][i])

                            item_dct = {
                                'scale': x_d_i_info['scale'][i].cpu().numpy().astype(np.float32),
                                'R_d': R_d_i.cpu().numpy().astype(np.float32),
                                'exp': x_d_i_info['exp'][i].reshape(1, 21, -1).cpu().numpy().astype(np.float32),
                                't': x_d_i_info['t'][i].cpu().numpy().astype(np.float32),
                            }

                            template_dct['motion'].append(item_dct)

                        I_p_lst = []
                        R_d_0, x_d_0_info = None, None

                        report_time_usage("Before animation Start")
                        for i in range(num_frame):
                            x_d_i_info = template_dct['motion'][i]
                            for key in x_d_i_info:
                                x_d_i_info[key] = torch.tensor(x_d_i_info[key]).cuda()
                            R_d_i = x_d_i_info['R_d']

                            if i == 0:
                                R_d_0 = R_d_i
                                x_d_0_info = x_d_i_info

                            if rt_inferencer.inf_cfg.flag_relative_motion:
                                R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s
                                delta_new = x_s_info['exp'].reshape(1, 21, -1) + (x_d_i_info['exp'] - x_d_0_info['exp'])
                                scale_new = x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
                                t_new = x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
                            else:
                                R_new = R_d_i
                                delta_new = x_d_i_info['exp']
                                scale_new = x_s_info['scale']
                                t_new = x_d_i_info['t']

                            t_new[..., 2].fill_(0)
                            x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

                            out = rt_inferencer.live_portrait_pipeline.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
                            I_p_i = rt_inferencer.live_portrait_pipeline.live_portrait_wrapper.parse_output(out['out'])[0]
                            I_p_lst.append(I_p_i)
                            
                        report_time_usage("Animation")

                        rt_inferencer.stream_new_animation(I_p_lst)
                        
                        if not rt_inferencer.process_display_queue():
                            break
                        
                        if chunk_counter >= 5:
                            break
                except Exception as e:
                    print("Exception in main loop:", e)
                    import traceback
                    traceback.print_exc()
                    break
            
            rt_inferencer._stop_display = True
            cv2.destroyAllWindows()