"""
Generic video recording using matplotlib figures and ffmpeg.
"""
import matplotlib.pyplot as plt
import subprocess
import tempfile
import os
import numpy as np
from PIL import Image

def _ensure_even_dimensions(image_path: str) -> None:
    """
    Ensure image has even width and height for H.264 compatibility.
    Pads with white pixels if dimensions are odd.
    
    Args:
        image_path: Path to PNG image file (modified in place)
    """
    img = Image.open(image_path)
    width, height = img.size
    
    # Check if padding needed
    new_width = width + (width % 2)
    new_height = height + (height % 2)    
    if new_width != width or new_height != height:
        # Create bigger image and paste original in
        new_img = Image.new('RGBA', (new_width, new_height), (255, 255, 255, 255))
        new_img.paste(img, (0, 0))
        new_img.save(image_path)

class VideoRecorder:
    """
    Generic video recorder for matplotlib figures using ffmpeg.
    Saves frames and creates MP4 video with H.264 codec.
    """
    
    def __init__(self, video_path: str, fps: int = 30, dpi: int = 100):
        """
        Initialize video recorder.
        
        Args:
            video_path: Full path to output MP4 file
            fps: Frame rate for video output
            dpi: DPI for frame capture
        """
        self.video_path = video_path
        self.temp_dir = None
        self.frame_files = []
        self.fps = fps
        self.dpi = dpi
        self.frame_count = 0
        
    def open_video(self):
        """
        Initialize video recording by creating temporary directory for frames.
        """
        output_dir = os.path.dirname(self.video_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create temporary directory for frames. Note: not auto-deleted by python.
        self.temp_dir = tempfile.mkdtemp(prefix='video_frames_')
        self.frame_files = []
        self.frame_count = 0
        
        print(f"Video recording initialized: {self.video_path}")
    
    def write_frame(self, fig=None):
        """
        Capture current matplotlib figure and save as frame.
        Args:
            fig: Matplotlib figure to capture (uses current figure if None)
        """
        if self.temp_dir is None:
            raise RuntimeError("Call open_video() first")
            
        if fig is None:
            fig = plt.gcf()
        
        # Save frame as PNG
        frame_file = os.path.join(self.temp_dir, f'frame_{self.frame_count:06d}.png')
        fig.savefig(frame_file, bbox_inches='tight', pad_inches=0, dpi=self.dpi)
        
        # Ensure even height dimensions for H.264 compatibility
        _ensure_even_dimensions(frame_file)
        
        self.frame_files.append(frame_file)
        self.frame_count += 1
    
    def close_video(self):
        """
        Create video from saved frames using ffmpeg and cleanup.
        """
        if self.temp_dir is None or not self.frame_files:
            print("No frames to create video")
            return
            
        try:
            # Create video with ffmpeg
            cmd = [
                'ffmpeg', '-y', '-r', str(self.fps),
                '-i', os.path.join(self.temp_dir, 'frame_%06d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                self.video_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Video created: {self.video_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg error: {e.stderr.decode()}")
            raise RuntimeError(f"Failed to create video: {e}")
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found. Please install ffmpeg.")
        finally:
            # cleanup
            for frame_file in self.frame_files:
                if os.path.exists(frame_file):
                    os.remove(frame_file)
            if os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)
            self.temp_dir = None
    
    def __del__(self):
        # cleanup
        if hasattr(self, 'temp_dir') and self.temp_dir and os.path.exists(self.temp_dir):
            for frame_file in self.frame_files:
                if os.path.exists(frame_file):
                    os.remove(frame_file)
            if os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)




