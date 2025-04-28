import os
import sys
import time
import random
import platform
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Scale
from PIL import Image, ImageEnhance, ImageFilter, ImageTk, ImageSequence
from io import BytesIO
import threading
from datetime import datetime, timedelta
import json
import subprocess
import tempfile
import shutil
import queue
import threading
try:
    import numpy as np
    HAS_NUMPY = True
    try:
        import skimage.color
        HAS_SKIMAGE = True
    except ImportError:
        HAS_SKIMAGE = False
        print("skimage not installed. Some advanced color features will be disabled.")
except ImportError:
    HAS_NUMPY = False
    HAS_SKIMAGE = False
    print("numpy not installed. Some advanced features will be disabled.")


class ThreadSafeQueue:
    """A thread-safe queue with additional management features"""
    def __init__(self, max_size=10):
        self._queue = queue.Queue(maxsize=max_size)
        self._cancel_event = threading.Event()
    
    def put(self, item, block=True, timeout=None):
        """Put an item in the queue with optional blocking and timeout"""
        try:
            self._queue.put(item, block=block, timeout=timeout)
        except queue.Full:
            # If queue is full, clear oldest items
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            # Retry putting the item
            self._queue.put(item, block=False)
    
    def get(self, block=True, timeout=None):
        """Get an item from the queue"""
        return self._queue.get(block=block, timeout=timeout)
    
    def task_done(self):
        """Mark a task as done"""
        self._queue.task_done()
    
    def cancel_pending_tasks(self):
        """Cancel any pending tasks"""
        self._cancel_event.set()
    
    def reset_cancel(self):
        """Reset the cancel event"""
        self._cancel_event.clear()
    
    def is_cancelled(self):
        """Check if tasks have been cancelled"""
        return self._cancel_event.is_set()
    
    def clear(self):
        """Clear all items from the queue"""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


class RobustThreadManager:
    """Manage multiple threads with cancellation and error handling"""
    def __init__(self, max_threads=5):
        self._semaphore = threading.Semaphore(max_threads)
        self._threads = []
        self._lock = threading.Lock()
    
    def submit(self, target, *args, **kwargs):
        """Submit a task to be run in a thread"""
        # Create a wrapper to manage thread lifecycle
        def thread_wrapper():
            try:
                # Acquire semaphore before running
                self._semaphore.acquire()
                target(*args, **kwargs)
            except Exception as e:
                print(f"Thread error: {e}")
            finally:
                # Always release semaphore
                self._semaphore.release()
        
        # Create and start thread
        thread = threading.Thread(target=thread_wrapper, daemon=True)
        
        # Add thread to managed list
        with self._lock:
            self._threads.append(thread)
        
        # Start thread
        thread.start()
        return thread
    
    def wait_all(self, timeout=None):
        """Wait for all threads to complete"""
        for thread in self._threads:
            thread.join(timeout)
        
        # Clean up completed threads
        with self._lock:
            self._threads = [t for t in self._threads if t.is_alive()]
    
    def cancel_all(self):
        """Attempt to cancel all running threads"""
        with self._lock:
            for thread in self._threads:
                if thread.is_alive():
                    # Ideally, threads should be designed to check for cancellation
                    try:
                        # For threads with cancellation support
                        thread._cancel_event.set()
                    except AttributeError:
                        # Fallback for threads without explicit cancellation
                        pass
            
            # Wait a short time for threads to terminate
            self.wait_all(timeout=1)


if getattr(sys, 'frozen', False):
    # If running as a bundled executable
    application_path = sys._MEIPASS
else:
    # If running as a script
    application_path = os.path.dirname(os.path.abspath(__file__))

FFMPEG_PATH = os.path.join(application_path, "ffmpeg.exe")
FFPROBE_PATH = os.path.join(application_path, "ffprobe.exe")

# Check if the file exists before proceeding
if not os.path.exists(FFMPEG_PATH):
    print(f"Warning: FFmpeg not found at {FFMPEG_PATH}")
if not os.path.exists(FFPROBE_PATH):
    print(f"Warning: FFprobe not found at {FFPROBE_PATH}")


class VideoPreviewFrame(tk.Frame):
    """Frame for video preview with play/pause controls"""
    def __init__(self, parent, video_path, ffmpeg_path, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent
        self.video_path = video_path
        self.ffmpeg_path = ffmpeg_path
        self.temp_dir = tempfile.mkdtemp(prefix="video_preview_")
        
        self.is_playing = False
        self.current_frame = 0
        self.frames = []
        self.play_speed = 1.0  # 1.0 = normal speed
        self.video_duration = 0  # Video duration in seconds
        self.selection_start = 0  # Start time of selection in seconds
        self.selection_end = 0    # End time of selection in seconds
        self.dragging = None      # Which selector is being dragged (left or right)
        self._timeline_events_bound = False
        
        # Output type selection
        self.output_mode_var = tk.StringVar(value="gif")  # Default to GIF mode
        
        # Video output parameters
        self.video_codec_var = tk.StringVar(value="h264")
        self.video_bitrate_var = tk.StringVar(value="1M")
        self.video_format_var = tk.StringVar(value="mp4")
        
        # Create UI elements
        self.preview_label = ttk.Label(self)
        self.preview_label.pack(padx=10, pady=10)
        
        # Control buttons frame
        controls_frame = ttk.Frame(self)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Play/Pause button
        self.play_button = ttk.Button(controls_frame, text="Play", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        # Speed control
        speed_frame = ttk.Frame(controls_frame)
        speed_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT)
        speed_values = ["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"]
        self.speed_var = tk.StringVar(value="1.0x")
        speed_combo = ttk.Combobox(speed_frame, textvariable=self.speed_var, 
                                 values=speed_values, width=5, state="readonly")
        speed_combo.pack(side=tk.LEFT, padx=5)
        speed_combo.bind("<<ComboboxSelected>>", self.change_speed)
        
        # Set in/out points buttons
        set_points_frame = ttk.Frame(controls_frame)
        set_points_frame.pack(side=tk.RIGHT, padx=10)
        
        ttk.Button(set_points_frame, text="Set In Point", 
                 command=self.set_in_point).pack(side=tk.LEFT, padx=5)
        ttk.Button(set_points_frame, text="Set Out Point", 
                 command=self.set_out_point).pack(side=tk.LEFT, padx=5)
        ttk.Button(set_points_frame, text="Reset Selection", 
                 command=self.reset_selection).pack(side=tk.LEFT, padx=5)
                 
        # Timeline frame
        timeline_frame = ttk.Frame(self)
        timeline_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Current time display
        self.time_display = ttk.Label(timeline_frame, text="00:00.000")
        self.time_display.pack(side=tk.TOP, anchor=tk.E, padx=5, pady=2)
        
        # Progress bar with timeline
        progress_frame = ttk.Frame(timeline_frame)
        progress_frame.pack(fill=tk.X, expand=True)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Scale(progress_frame, variable=self.progress_var, 
                                    from_=0, to=100, orient=tk.HORIZONTAL,
                                    command=self.seek)
        self.progress_bar.pack(fill=tk.X, expand=True)
        
        # Timeline markers
        self.timeline_canvas = tk.Canvas(progress_frame, height=20, bg='white', highlightthickness=0)
        self.timeline_canvas.pack(fill=tk.X, expand=True, pady=2)
        
        # Status label
        self.status_var = tk.StringVar(value="Loading preview...")
        status_label = ttk.Label(self, textvariable=self.status_var)
        status_label.pack(pady=5)
        
        # Extract preview frames
        self.extract_frames()
    
    def update_selection(self, start_time, end_time):
        """Update the timeline selection based on start and end times"""
        # Ensure times are within video duration
        start_time = max(0, min(start_time, self.video_duration))
        end_time = max(start_time, min(end_time, self.video_duration))
        
        # Update selection
        self.selection_start = start_time
        self.selection_end = end_time
        
        # Redraw selection indicators
        self.draw_selection_indicators()
        
        # Adjust current frame if necessary
        if hasattr(self, 'current_frame') and len(self.frames) > 0:
            current_time = (self.current_frame / len(self.frames)) * self.video_duration
            if current_time < start_time:
                new_frame = int((start_time / self.video_duration) * len(self.frames))
                self.current_frame = max(0, min(new_frame, len(self.frames) - 1))
                self.preview_label.config(image=self.frames[self.current_frame])
                self.progress_var.set(self.current_frame)
            elif current_time > end_time:
                new_frame = int((end_time / self.video_duration) * len(self.frames))
                self.current_frame = max(0, min(new_frame, len(self.frames) - 1))
                self.preview_label.config(image=self.frames[self.current_frame])
                self.progress_var.set(self.current_frame)
                
    def draw_selection_indicators(self):
        """Trigger timeline redraw with current selection in a thread-safe way"""
        if hasattr(self, 'timeline_canvas'):
            # Use after() to ensure timeline is updated on the main thread
            self.after(0, self.draw_timeline_markers)
     
    def extract_frames(self):
        """Extract frames from video for preview"""
        try:
            # Get video duration first
            try:
                duration_cmd = [
                    FFPROBE_PATH,
                    '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    self.video_path
                ]
                
                result = subprocess.run(duration_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.stdout.strip():
                    self.video_duration = float(result.stdout.strip())
                else:
                    # If no duration found, set a default and log the error
                    self.video_duration = 10.0
                    print(f"Could not determine video duration. Error: {result.stderr}")
                    self.status_var.set("Warning: Could not determine video duration")
            except Exception as e:
                # Set a default duration if ffprobe fails
                self.video_duration = 10.0
                print(f"Error getting video duration: {str(e)}")
                self.status_var.set("Warning: Could not determine video duration")
            
            # Clear old frames
            for file in os.listdir(self.temp_dir):
                if file.startswith("preview_") and file.endswith(".jpg"):
                    os.remove(os.path.join(self.temp_dir, file))
            
            # Get video info for preview frames
            cmd = [
                self.ffmpeg_path,
                '-i', self.video_path,
                '-vf', 'fps=10,scale=400:-1',  # 10fps, resize to 400px width
                os.path.join(self.temp_dir, 'preview_%04d.jpg')
            ]
            
            self.status_var.set("Extracting preview frames...")
            
            # Run in a separate thread
            threading.Thread(target=self._run_extraction, args=(cmd,), daemon=True).start()
            
            # Draw timeline markers
            self.after(1000, self.draw_timeline_markers)  # Wait a second to make sure video info is loaded
            
        except Exception as e:
            self.status_var.set(f"Error preparing preview: {str(e)}")
            print(f"Error in extract_frames: {str(e)}")
    
    def _run_extraction(self, cmd):
        """Run the frame extraction process"""
        try:
            # Execute FFmpeg command
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                # FFmpeg reported an error
                error_message = stderr.decode('utf-8', errors='ignore') if stderr else "Unknown error"
                self.status_var.set(f"Error extracting frames: {error_message[:100]}...")
                print(f"FFmpeg error: {error_message}")
                return
            
            # Load extracted frames
            self.frames = []
            frame_files = sorted([f for f in os.listdir(self.temp_dir) 
                              if f.startswith("preview_") and f.endswith(".jpg")])
            
            if not frame_files:
                self.status_var.set("No frames could be extracted")
                print("No frame files were found after FFmpeg execution")
                return
                
            for frame_file in frame_files:
                try:
                    img_path = os.path.join(self.temp_dir, frame_file)
                    img = Image.open(img_path)
                    photo = ImageTk.PhotoImage(img)
                    self.frames.append(photo)
                except Exception as e:
                    print(f"Error loading frame {frame_file}: {str(e)}")
            
            if self.frames:
                self.current_frame = 0
                self.preview_label.config(image=self.frames[0])
                self.status_var.set(f"Ready to play ({len(self.frames)} frames)")
                self.progress_var.set(0)
                self.progress_bar.config(to=len(self.frames)-1)
                
                # Set selection to full range initially
                self.selection_start = 0
                self.selection_end = self.video_duration
                
                # Update timeline markers
                self.draw_timeline_markers()
            else:
                self.status_var.set("No frames could be extracted")
                
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            print(f"Error in _run_extraction: {str(e)}")
    
    def toggle_play(self):
        """Toggle play/pause"""
        if self.is_playing:
            self.is_playing = False
            self.play_button.config(text="Play")
        else:
            self.is_playing = True
            self.play_button.config(text="Pause")
            
            # If we're at or beyond the end boundary, restart from the start boundary
            if hasattr(self, 'video_duration') and self.video_duration > 0 and len(self.frames) > 0:
                current_time = (self.current_frame / len(self.frames)) * self.video_duration
                if hasattr(self, 'selection_end') and self.selection_end > 0 and current_time >= self.selection_end:
                    # Jump to start boundary
                    if hasattr(self, 'selection_start'):
                        frame_idx = int((self.selection_start / self.video_duration) * len(self.frames))
                        self.current_frame = max(0, min(frame_idx, len(self.frames) - 1))
                        self.preview_label.config(image=self.frames[self.current_frame])
                        self.progress_var.set(self.current_frame)
            
            # Start playback
            self._play_next_frame()
    
    def _play_next_frame(self):
        """Play the next frame"""
        try:
            # Basic validation
            if not self.is_playing:
                return
            
            if not self.frames or len(self.frames) == 0:
                print("No frames available - stopping playback")
                self.is_playing = False
                self.play_button.config(text="Play")
                return
            
            # Calculate current time position
            current_time = 0
            if hasattr(self, 'video_duration') and self.video_duration > 0:
                current_time = (self.current_frame / len(self.frames)) * self.video_duration
            
            # Check if we've reached the end boundary
            if hasattr(self, 'selection_end') and self.selection_end > 0:
                if current_time >= self.selection_end:
                    # We've reached the end of the selection - loop back to start
                    if hasattr(self, 'selection_start'):
                        frame_idx = int((self.selection_start / self.video_duration) * len(self.frames))
                        self.current_frame = max(0, min(frame_idx, len(self.frames) - 1))
                    else:
                        # No start bound defined, use 0
                        self.current_frame = 0
                    
                    # Update display
                    if self.current_frame < len(self.frames):
                        self.preview_label.config(image=self.frames[self.current_frame])
                        self.progress_var.set(self.current_frame)
                else:
                    # Not at end yet, just go to next frame
                    self.current_frame = (self.current_frame + 1) % len(self.frames)
            else:
                # No end bound defined, just loop through all frames
                self.current_frame = (self.current_frame + 1) % len(self.frames)
        
            # Display the current frame (with bounds check)
            if 0 <= self.current_frame < len(self.frames):
                self.preview_label.config(image=self.frames[self.current_frame])
                self.progress_var.set(self.current_frame)
            else:
                print(f"Invalid frame index: {self.current_frame}, max: {len(self.frames)-1}")
                self.current_frame = 0  # Reset to valid frame
                self.preview_label.config(image=self.frames[self.current_frame])
                self.progress_var.set(self.current_frame)
        
            # Update time display
            if hasattr(self, 'video_duration') and self.video_duration > 0:
                current_time = (self.current_frame / len(self.frames)) * self.video_duration
                mins = int(current_time // 60)
                secs = current_time % 60
                self.time_display.config(text=f"{mins:02d}:{secs:06.3f}")
        
            # Schedule next frame based on speed
            play_speed = getattr(self, 'play_speed', 1.0)  # Default to 1.0 if not set
            delay = int(100 / float(play_speed))  # Base delay is 100ms (10fps)
            self.after(delay, self._play_next_frame)
        
        except Exception as e:
            print(f"Error in _play_next_frame: {str(e)}")
            self.is_playing = False
            self.play_button.config(text="Play")
    
    def change_speed(self, event=None):
        """Change playback speed"""
        speed_text = self.speed_var.get().replace('x', '')
        try:
            self.play_speed = float(speed_text)
        except ValueError:
            self.play_speed = 1.0
    
    def seek(self, value):
        """Seek to a specific frame"""
        if not self.frames:
            return
            
        try:
            frame_idx = int(float(value))
            if 0 <= frame_idx < len(self.frames):
                # Calculate the time position for this frame
                if hasattr(self, 'video_duration') and self.video_duration > 0:
                    current_time = (frame_idx / len(self.frames)) * self.video_duration
                    
                    # Check if within selection boundaries
                    if (hasattr(self, 'selection_start') and current_time < self.selection_start):
                        # Trying to seek before start boundary - snap to start
                        frame_idx = int((self.selection_start / self.video_duration) * len(self.frames))
                    elif (hasattr(self, 'selection_end') and self.selection_end > 0 and 
                          current_time > self.selection_end):
                        # Trying to seek beyond end boundary - snap to end
                        frame_idx = int((self.selection_end / self.video_duration) * len(self.frames))
                
                # Update frame display
                self.current_frame = frame_idx
                self.preview_label.config(image=self.frames[self.current_frame])
                
                # Update time display
                if hasattr(self, 'video_duration') and self.video_duration > 0:
                    current_time = (frame_idx / len(self.frames)) * self.video_duration
                    mins = int(current_time // 60)
                    secs = current_time % 60
                    self.time_display.config(text=f"{mins:02d}:{secs:06.3f}")
                    
                    # Notify parent of time update if method exists
                    if hasattr(self.parent, 'update_current_time') and callable(getattr(self.parent, 'update_current_time')):
                        self.parent.update_current_time(current_time)
        except (ValueError, IndexError):
            pass
    
    def draw_timeline_markers(self):
        """Draw timeline markers with frame timestamps and interactive selection"""
        if not hasattr(self, 'video_duration') or self.video_duration <= 0 or not self.frames:
            return
            
        # Clear canvas
        self.timeline_canvas.delete("all")
        
        # Draw timeline
        width = self.timeline_canvas.winfo_width()
        if width <= 1:  # Not yet properly sized
            self.timeline_canvas.update_idletasks()
            width = self.timeline_canvas.winfo_width()
            if width <= 1:  # Still not sized, use parent width
                width = self.winfo_width() - 20
                
        height = self.timeline_canvas.winfo_height()
        
        # Draw base line
        self.timeline_canvas.create_line(0, height/2, width, height/2, fill="gray")
        
        # Draw time markers
        marker_count = 10  # Number of markers to show
        interval = self.video_duration / marker_count
        
        for i in range(marker_count + 1):
            x_pos = (i / marker_count) * width
            time_val = i * interval
            
            # Draw marker
            self.timeline_canvas.create_line(x_pos, 0, x_pos, height/2, fill="black")
            
            # Draw time label
            mins = int(time_val // 60)
            secs = time_val % 60
            time_str = f"{mins}:{secs:02.0f}"
            self.timeline_canvas.create_text(x_pos, height*0.8, text=time_str, font=("Arial", 7))
        
        # Ensure default selection if not set
        if not hasattr(self, 'selection_start'):
            self.selection_start = 0
        if not hasattr(self, 'selection_end') or self.selection_end <= 0:
            self.selection_end = self.video_duration
        
        # Calculate current selection positions
        start_pos = (self.selection_start / self.video_duration) * width
        end_pos = (self.selection_end / self.video_duration) * width
        
        # Draw selection range (shaded area)
        self.timeline_canvas.create_rectangle(
            start_pos, 0, end_pos, height, 
            fill="lightblue", outline="", stipple="gray50", tags="selection"
        )
        
        # Draw non-selected areas (darkened)
        self.timeline_canvas.create_rectangle(
            0, 0, start_pos, height,
            fill="gray", outline="", stipple="gray25", tags="selection"
        )
        self.timeline_canvas.create_rectangle(
            end_pos, 0, width, height,
            fill="gray", outline="", stipple="gray25", tags="selection"
        )
        
        # Draw start selector (left)
        handle_width = 6
        handle_height = height
        
        start_handle = self.timeline_canvas.create_rectangle(
            start_pos-handle_width/2, 0, 
            start_pos+handle_width/2, handle_height, 
            fill="blue", outline="white", width=1, tags="start_selector"
        )
        
        # Draw end selector (right)
        end_handle = self.timeline_canvas.create_rectangle(
            end_pos-handle_width/2, 0, 
            end_pos+handle_width/2, handle_height, 
            fill="red", outline="white", width=1, tags="end_selector"
        )
        
        # Add selection time labels
        start_time_str = self.format_time(self.selection_start)
        end_time_str = self.format_time(self.selection_end)
        
        # Display start time
        self.timeline_canvas.create_text(
            start_pos, height-5,
            text=start_time_str, fill="blue", font=("Arial", 7),
            anchor="s", tags="selection_text"
        )
        
        # Display end time
        self.timeline_canvas.create_text(
            end_pos, height-5,
            text=end_time_str, fill="red", font=("Arial", 7),
            anchor="s", tags="selection_text"
        )
        
        # Make sure selectors are on top
        self.timeline_canvas.tag_raise("start_selector")
        self.timeline_canvas.tag_raise("end_selector")
        self.timeline_canvas.tag_raise("selection_text")
        
        # Bind mouse events for range selection
        self.timeline_canvas.bind("<ButtonPress-1>", self.on_selector_press)
        self.timeline_canvas.bind("<B1-Motion>", self.on_selector_drag)
        self.timeline_canvas.bind("<ButtonRelease-1>", self.on_selector_release)
    
    def format_time(self, seconds):
        """Format time in MM:SS format"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
    
    def on_selector_press(self, event):
        """Handle mouse press on timeline selectors"""
        x, y = event.x, event.y
        width = self.timeline_canvas.winfo_width()
        
        # Calculate positions
        start_pos = (self.selection_start / self.video_duration) * width if hasattr(self, 'selection_start') else 0
        end_pos = (self.selection_end / self.video_duration) * width if hasattr(self, 'selection_end') and self.selection_end > 0 else width
        
        # Determine if start or end selector was clicked
        if abs(x - start_pos) <= 10:  # Within 10 pixels of start selector
            self.dragging = "start"
            # Update cursor
            self.timeline_canvas.config(cursor="sb_h_double_arrow")
        elif abs(x - end_pos) <= 10:  # Within 10 pixels of end selector
            self.dragging = "end"
            # Update cursor
            self.timeline_canvas.config(cursor="sb_h_double_arrow")
        elif start_pos <= x <= end_pos:  # Within selection area
            self.dragging = "move"
            self.drag_start_x = x
            self.drag_start_selection = (self.selection_start, self.selection_end)
            # Update cursor
            self.timeline_canvas.config(cursor="fleur")
        else:
            self.dragging = None
            # Reset cursor
            self.timeline_canvas.config(cursor="")
    
    def on_selector_drag(self, event):
        """Handle drag to adjust selection range"""
        if self.dragging is None or not hasattr(self, 'video_duration') or self.video_duration <= 0:
                return
                
        width = self.timeline_canvas.winfo_width()
        x = max(0, min(event.x, width))  # Constrain to timeline width
        
        # Convert position to time
        time_value = (x / width) * self.video_duration
        
        if self.dragging == "start":
                # Update start time (can't go beyond end time)
                self.selection_start = min(time_value, self.selection_end if hasattr(self, 'selection_end') and self.selection_end > 0 else self.video_duration)
                
                # If play position is now before start, move it to start
                if hasattr(self, 'current_frame') and len(self.frames) > 0:
                        current_time = (self.current_frame / len(self.frames)) * self.video_duration
                        if current_time < self.selection_start:
                                new_frame = int((self.selection_start / self.video_duration) * len(self.frames))
                                self.current_frame = max(0, min(new_frame, len(self.frames) - 1))
                                self.preview_label.config(image=self.frames[self.current_frame])
                                self.progress_var.set(self.current_frame)
        
        elif self.dragging == "end":
                # Update end time (can't go before start time)
                self.selection_end = max(time_value, self.selection_start if hasattr(self, 'selection_start') else 0)
                
                # If play position is now beyond end, move it to end
                if hasattr(self, 'current_frame') and len(self.frames) > 0:
                        current_time = (self.current_frame / len(self.frames)) * self.video_duration
                        if current_time > self.selection_end:
                                new_frame = int((self.selection_end / self.video_duration) * len(self.frames))
                                self.current_frame = max(0, min(new_frame, len(self.frames) - 1))
                                self.preview_label.config(image=self.frames[self.current_frame])
                                self.progress_var.set(self.current_frame)
                        
        elif self.dragging == "move" and hasattr(self, 'drag_start_x') and hasattr(self, 'drag_start_selection'):
                # Calculate movement delta in time
                delta_x = x - self.drag_start_x
                delta_time = (delta_x / width) * self.video_duration
                
                # Get original selection points
                orig_start, orig_end = self.drag_start_selection
                
                # Calculate new positions while keeping selection width the same
                new_start = max(0, min(orig_start + delta_time, self.video_duration - (orig_end - orig_start)))
                new_end = min(self.video_duration, new_start + (orig_end - orig_start))
                
                # Update selection
                self.selection_start = new_start
                self.selection_end = new_end
                
                # Adjust current frame position to stay within new selection
                if hasattr(self, 'current_frame') and len(self.frames) > 0:
                        current_time = (self.current_frame / len(self.frames)) * self.video_duration
                        if current_time < new_start:
                                new_frame = int((new_start / self.video_duration) * len(self.frames))
                                self.current_frame = max(0, min(new_frame, len(self.frames) - 1))
                        elif current_time > new_end:
                                new_frame = int((new_end / self.video_duration) * len(self.frames))
                                self.current_frame = max(0, min(new_frame, len(self.frames) - 1))
                        
                        # Update display
                        self.preview_label.config(image=self.frames[self.current_frame])
                        self.progress_var.set(self.current_frame)
        
        # Update parent app if the method exists
        if hasattr(self.parent, 'update_time_selection') and callable(getattr(self.parent, 'update_time_selection')):
                self.parent.update_time_selection(self.selection_start, self.selection_end)
        
        # Update the timeline display
        self.draw_selection_indicators()
    
    def on_selector_release(self, event):
        """Handle release after selector drag"""
        # Update selection boundaries for video playback
        if self.dragging == "start" or self.dragging == "end" or self.dragging == "move":
            # If we're playing, make sure we're within bounds
            if self.is_playing and hasattr(self, 'current_frame') and len(self.frames) > 0:
                current_time = (self.current_frame / len(self.frames)) * self.video_duration
                if current_time < self.selection_start:
                    # Snap to start
                    new_frame = int((self.selection_start / self.video_duration) * len(self.frames))
                    self.current_frame = max(0, min(new_frame, len(self.frames) - 1))
                    self.preview_label.config(image=self.frames[self.current_frame])
                elif current_time > self.selection_end:
                    # Snap to start (loop back)
                    new_frame = int((self.selection_start / self.video_duration) * len(self.frames))
                    self.current_frame = max(0, min(new_frame, len(self.frames) - 1))
                    self.preview_label.config(image=self.frames[self.current_frame])
        
        # Reset dragging state and cursor
        self.dragging = None
        self.timeline_canvas.config(cursor="")
        
        # Clear any drag start information
        if hasattr(self, 'drag_start_x'):
            delattr(self, 'drag_start_x')
        if hasattr(self, 'drag_start_selection'):
            delattr(self, 'drag_start_selection')
        
        # Notify parent of final selection
        if hasattr(self.parent, 'update_time_selection') and callable(getattr(self.parent, 'update_time_selection')):
            self.parent.update_time_selection(self.selection_start, self.selection_end)
    
    def set_in_point(self):
        """Set the in point (start boundary) to the current frame position"""
        if not self.frames or not hasattr(self, 'video_duration') or self.video_duration <= 0:
            return
            
        # Calculate current time position
        current_time = (self.current_frame / len(self.frames)) * self.video_duration
        
        # Set as start point (but don't go beyond end point)
        if hasattr(self, 'selection_end') and self.selection_end > 0:
            self.selection_start = min(current_time, self.selection_end - 0.1)  # Ensure at least 0.1s selection
        else:
            self.selection_start = current_time
        
        # Update timeline display
        self.draw_selection_indicators()
        
        # Notify parent if necessary
        if hasattr(self.parent, 'update_time_selection') and callable(getattr(self.parent, 'update_time_selection')):
            self.parent.update_time_selection(self.selection_start, self.selection_end)
    
    def set_out_point(self):
        """Set the out point (end boundary) to the current frame position"""
        if not self.frames or not hasattr(self, 'video_duration') or self.video_duration <= 0:
            return
            
        # Calculate current time position
        current_time = (self.current_frame / len(self.frames)) * self.video_duration
        
        # Set as end point (but don't go before start point)
        if hasattr(self, 'selection_start'):
            self.selection_end = max(current_time, self.selection_start + 0.1)  # Ensure at least 0.1s selection
        else:
            self.selection_end = current_time
        
        # Update timeline display
        self.draw_selection_indicators()
        
        # Notify parent if necessary
        if hasattr(self.parent, 'update_time_selection') and callable(getattr(self.parent, 'update_time_selection')):
            self.parent.update_time_selection(self.selection_start, self.selection_end)
    
    def reset_selection(self):
        """Reset the selection to the entire video"""
        if not hasattr(self, 'video_duration') or self.video_duration <= 0:
            return
            
        # Reset selection to full video
        self.selection_start = 0
        self.selection_end = self.video_duration
        
        # Update timeline display
        self.draw_selection_indicators()
        
        # Notify parent if necessary
        if hasattr(self.parent, 'update_time_selection') and callable(getattr(self.parent, 'update_time_selection')):
            self.parent.update_time_selection(self.selection_start, self.selection_end)
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Error cleaning up preview: {str(e)}")


class AdvancedGifWasher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Advanced GIF Washer")
        self.root.geometry("900x750")
        
        # Initialize variables
        self.input_path = None
        self.frames = []
        self.processed_frames = []
        self.current_frame_index = 0
        self.is_processing = False
        self.is_video = False
        self.video_duration = 0
        self.temp_dir = None
        self.frame_rate = 30
        self.video_preview = None
        self.is_folder_mode = False  # Added for folder mode tracking
        self.input_folder = None     # Added to store folder path
        
        self.current_segment_start = None
        self.current_segment_end = None
        
        # Add thread management
        self.thread_manager = RobustThreadManager()
        
        # Create output folders
        self.setup_folders()
        
        # Time selection variables
        self.start_time_var = tk.DoubleVar(value=0.0)
        self.end_time_var = tk.DoubleVar(value=5.0)
        self.duration_var = tk.DoubleVar(value=5.0)
        
        # GIF parameters
        self.frame_skip_var = tk.IntVar(value=1)
        self.frame_duration_var = tk.DoubleVar(value=1.0)  # Default to 1.0x (original speed)
        self.quality_var = tk.IntVar(value=90)
        self.max_width_var = tk.IntVar(value=500)
        self.max_gif_duration_var = tk.DoubleVar(value=5.0)
        # Output type selection
        self.output_mode_var = tk.StringVar(value="gif")  # Default to GIF mode
        
        # Video output parameters
        self.video_codec_var = tk.StringVar(value="h264")
        self.video_bitrate_var = tk.StringVar(value="1M")
        self.video_format_var = tk.StringVar(value="mp4")
        
        
        # Auto-generation parameters
        self.auto_gen_enabled = tk.BooleanVar(value=False)
        self.num_gifs_var = tk.IntVar(value=3)
        self.min_gif_length_var = tk.DoubleVar(value=1.0)
        self.max_gif_length_var = tk.DoubleVar(value=5.0)
        
        # Effect parameters
        self.brightness_min_var = tk.DoubleVar(value=0.8)
        self.brightness_max_var = tk.DoubleVar(value=1.2)
        self.contrast_min_var = tk.DoubleVar(value=0.8)
        self.contrast_max_var = tk.DoubleVar(value=1.2)
        self.saturation_min_var = tk.DoubleVar(value=0.8)
        self.saturation_max_var = tk.DoubleVar(value=1.2)
        self.blur_min_var = tk.DoubleVar(value=0)
        self.blur_max_var = tk.DoubleVar(value=0.5)
        self.rotation_min_var = tk.DoubleVar(value=-2)
        self.rotation_max_var = tk.DoubleVar(value=2)
        self.resize_min_var = tk.DoubleVar(value=0.95)
        self.resize_max_var = tk.DoubleVar(value=1.05)
        
        # Toggles
        self.brightness_enabled = tk.BooleanVar(value=True)
        self.contrast_enabled = tk.BooleanVar(value=True)
        self.saturation_enabled = tk.BooleanVar(value=True)
        self.blur_enabled = tk.BooleanVar(value=True)
        self.rotation_enabled = tk.BooleanVar(value=False)
        self.resize_enabled = tk.BooleanVar(value=True)
        
        # Consistency settings
        self.consistent_effects = tk.BooleanVar(value=True)
        self.effect_seed = tk.IntVar(value=random.randint(1, 1000000))
        
        # GIF options
        self.loop_var = tk.BooleanVar(value=True)
        self.reverse_var = tk.BooleanVar(value=False)
        self.ping_pong_var = tk.BooleanVar(value=False)
        
        # Status variable
        self.status_var = tk.StringVar(value="Ready. Select a video or GIF file to begin.")
        
        # Setup GUI
        self.setup_gui()
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_folders(self):
        """Create necessary folders for the application"""
        if getattr(sys, 'frozen', False):
            # If running as compiled executable
            application_path = os.path.dirname(sys.executable)
        else:
            # If running as script
            application_path = os.path.dirname(os.path.abspath(__file__))
        
        try:
            # Create folders
            self.gif_folder = os.path.join(application_path, "GifWashing")
            self.output_folder = os.path.join(self.gif_folder, "processed")
            
            # Create the folders if they don't exist
            if not os.path.exists(self.gif_folder):
                os.makedirs(self.gif_folder)
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
            
            # Create temp folder for video frames
            self.temp_dir = tempfile.mkdtemp(prefix="gifwash_")
            
        except Exception as e:
            print(f"Error creating folders: {str(e)}")
            messagebox.showerror("Error", f"Could not create folders: {str(e)}")
    
    def setup_gui(self):
        """Setup the application GUI"""
        # Create a main scrollable container
        # Main container frame
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create vertical scrollbar
        main_scrollbar = ttk.Scrollbar(main_container, orient=tk.VERTICAL)
        main_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create canvas for scrolling
        self.main_canvas = tk.Canvas(main_container, yscrollcommand=main_scrollbar.set)
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure scrollbar to scroll canvas
        main_scrollbar.config(command=self.main_canvas.yview)
        
        # Create a frame inside canvas to hold all content
        self.scrollable_frame = ttk.Frame(self.main_canvas)
        
        # Add scrollable frame to canvas
        self.canvas_frame = self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Configure canvas to resize with frame
        self.scrollable_frame.bind("<Configure>", self.on_frame_configure)
        self.main_canvas.bind("<Configure>", self.on_canvas_configure)
        
        # Bind mousewheel to scrolling
        self.bind_mousewheel()
        
        # Create a notebook (tabbed interface) inside the scrollable frame
        notebook = ttk.Notebook(self.scrollable_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.main_frame = ttk.Frame(notebook)
        self.effects_frame = ttk.Frame(notebook)
        self.auto_frame = ttk.Frame(notebook)
        
        notebook.add(self.main_frame, text="Main")
        notebook.add(self.effects_frame, text="Effects")
        notebook.add(self.auto_frame, text="Auto Generation")
        
        # Setup each tab
        self.setup_main_tab()
        self.setup_effects_tab()
        self.setup_auto_tab()
        
        # Progress bar at the bottom
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(self.scrollable_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)
        
        # Status label
        ttk.Label(self.scrollable_frame, textvariable=self.status_var).pack(pady=5)
        
        # Button frame
        button_frame = ttk.Frame(self.scrollable_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="Extract Random Segment", 
                command=self.extract_random_segment).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export Frames as Images", 
                command=self.export_frames_as_images).pack(side=tk.LEFT, padx=5)
    
    def bulk_convert_videos(self):
        """Convert multiple videos to GIFs in batch mode"""
        if self.is_processing:
            self.status_var.set("Processing already in progress")
            return
            
        # Ask user to select a folder containing videos
        folder_path = filedialog.askdirectory(
            title="Select Folder Containing Videos"
        )
        
        if not folder_path:
            return  # User cancelled
            
        # Ask for output settings
        settings_dialog = tk.Toplevel(self.root)
        settings_dialog.title("Bulk Conversion Settings")
        settings_dialog.geometry("500x500")
        settings_dialog.resizable(False, False)
        settings_dialog.transient(self.root)
        settings_dialog.grab_set()
        
        # Create settings widgets
        settings_frame = ttk.Frame(settings_dialog, padding=10)
        settings_frame.pack(fill=tk.BOTH, expand=True)
        
        # Method selection
        method_frame = ttk.LabelFrame(settings_frame, text="Conversion Method")
        method_frame.pack(fill=tk.X, pady=5)
        
        method_var = tk.StringVar(value="exact")
        ttk.Radiobutton(method_frame, text="Exact Copies (Same Segment for Each Video)", 
                      variable=method_var, value="exact").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(method_frame, text="Random Segments", 
                      variable=method_var, value="random").pack(anchor=tk.W, padx=5, pady=2)
        
        # Segment details for exact method
        exact_frame = ttk.LabelFrame(settings_frame, text="Exact Copy Settings")
        exact_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(exact_frame, text="Segment Start Time (seconds from start):").pack(anchor=tk.W, padx=5, pady=2)
        start_var = tk.DoubleVar(value=0.0)
        ttk.Entry(exact_frame, textvariable=start_var, width=10).pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Label(exact_frame, text="Segment Duration (seconds):").pack(anchor=tk.W, padx=5, pady=2)
        duration_var = tk.DoubleVar(value=3.0)
        ttk.Entry(exact_frame, textvariable=duration_var, width=10).pack(anchor=tk.W, padx=5, pady=2)
        
        # Random segment settings
        random_frame = ttk.LabelFrame(settings_frame, text="Random Segment Settings")
        random_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(random_frame, text="Min Segment Length (seconds):").pack(anchor=tk.W, padx=5, pady=2)
        min_length_var = tk.DoubleVar(value=1.0)
        ttk.Entry(random_frame, textvariable=min_length_var, width=10).pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Label(random_frame, text="Max Segment Length (seconds):").pack(anchor=tk.W, padx=5, pady=2)
        max_length_var = tk.DoubleVar(value=5.0)
        ttk.Entry(random_frame, textvariable=max_length_var, width=10).pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Label(random_frame, text="GIFs per Video:").pack(anchor=tk.W, padx=5, pady=2)
        gifs_per_video_var = tk.IntVar(value=1)
        ttk.Entry(random_frame, textvariable=gifs_per_video_var, width=10).pack(anchor=tk.W, padx=5, pady=2)
        
        # Shared GIF settings
        general_frame = ttk.LabelFrame(settings_frame, text="General GIF Settings")
        general_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(general_frame, text="Max Width (pixels):").pack(anchor=tk.W, padx=5, pady=2)
        max_width_var = tk.IntVar(value=400)
        ttk.Entry(general_frame, textvariable=max_width_var, width=10).pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Label(general_frame, text="Frame Skip:").pack(anchor=tk.W, padx=5, pady=2)
        frame_skip_var = tk.IntVar(value=2)
        ttk.Entry(general_frame, textvariable=frame_skip_var, width=10).pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Label(general_frame, text="Playback Speed:").pack(anchor=tk.W, padx=5, pady=2)
        speed_var = tk.DoubleVar(value=1.0)
        ttk.Entry(general_frame, textvariable=speed_var, width=10).pack(anchor=tk.W, padx=5, pady=2)
        
        # Quality settings
        quality_var = tk.IntVar(value=85)
        ttk.Label(general_frame, text="Quality (1-100):").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Entry(general_frame, textvariable=quality_var, width=10).pack(anchor=tk.W, padx=5, pady=2)
        
        # File inclusion filter
        filter_frame = ttk.LabelFrame(settings_frame, text="File Filters")
        filter_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(filter_frame, text="Process files containing (leave empty for all):").pack(anchor=tk.W, padx=5, pady=2)
        filter_var = tk.StringVar(value="")
        ttk.Entry(filter_frame, textvariable=filter_var, width=30).pack(anchor=tk.W, padx=5, pady=2)
        
        # Button frame
        button_frame = ttk.Frame(settings_dialog)
        button_frame.pack(fill=tk.X, pady=10)
        
        def on_cancel():
            settings_dialog.destroy()
            
        def on_start():
            # Hide the dialog
            settings_dialog.withdraw()
            
            # Start the bulk conversion process
            threading.Thread(target=self._perform_bulk_conversion, args=(
                folder_path,
                method_var.get(),
                start_var.get(),
                duration_var.get(),
                min_length_var.get(),
                max_length_var.get(),
                gifs_per_video_var.get(),
                max_width_var.get(),
                frame_skip_var.get(),
                speed_var.get(),
                quality_var.get(),
                filter_var.get()
            ), daemon=True).start()
            
            # Close the dialog
            settings_dialog.destroy()
            
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Start Bulk Conversion", command=on_start).pack(side=tk.RIGHT, padx=5)
        
        # Center dialog
        settings_dialog.update_idletasks()
        width = settings_dialog.winfo_width()
        height = settings_dialog.winfo_height()
        x = (settings_dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (settings_dialog.winfo_screenheight() // 2) - (height // 2)
        settings_dialog.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        
        # Wait for dialog to complete
        self.root.wait_window(settings_dialog)
    
    def update_duration(self):
        """Update the duration label based on start and end times"""
        try:
                # Calculate duration from start and end times
                start = self.start_time_var.get()
                end = self.end_time_var.get()
                duration = end - start
                
                # Update duration variable and label
                self.duration_var.set(duration)
                if hasattr(self, 'duration_label'):
                        self.duration_label.config(text=f"{duration:.1f}")
        except Exception as e:
                print(f"Error updating duration: {e}")
    
    def _perform_bulk_conversion(self, folder_path, method, start_time, segment_duration, 
                               min_length, max_length, gifs_per_video, max_width, 
                               frame_skip, playback_speed, quality, file_filter):
        """Process all videos in a folder"""
        try:
            self.is_processing = True
            
            # Temporarily store original settings
            original_max_width = self.max_width_var.get()
            original_frame_skip = self.frame_skip_var.get()
            original_speed = self.frame_duration_var.get()
            original_quality = self.quality_var.get()
            
            # Set new settings
            self.max_width_var.set(max_width)
            self.frame_skip_var.set(frame_skip)
            self.frame_duration_var.set(playback_speed)
            self.quality_var.set(quality)
            
            # Get list of video files
            video_files = []
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                
                # Check if it's a file and has a video extension
                if (os.path.isfile(file_path) and
                    file_name.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.mkv', '.flv'))):
                    
                    # Apply filter if specified
                    if not file_filter or file_filter.lower() in file_name.lower():
                        video_files.append(file_path)
            
            # Update status
            self.status_var.set(f"Found {len(video_files)} videos to process")
            self.progress_var.set(0)
            self.root.update_idletasks()
            
            # Process each video
            for i, video_path in enumerate(video_files):
                file_name = os.path.basename(video_path)
                self.status_var.set(f"Processing video {i+1}/{len(video_files)}: {file_name}")
                self.progress_var.set((i / len(video_files)) * 100)
                self.root.update_idletasks()
                
                # Set input path to current video
                self.input_path = video_path
                
                if method == "exact":
                    # Process with exact segment for all videos
                    end_time = start_time + segment_duration
                    
                    # Generate a new random seed for each video
                    self.effect_seed.set(random.randint(1, 1000000))
                    
                    try:
                        # Create GIF with direct method
                        self._create_single_gif_direct(
                            start_time,
                            end_time,
                            playback_speed,
                            f"bulk_{i+1}_"
                        )
                    except Exception as e:
                        print(f"Error processing video {file_name}: {e}")
                        # Continue with next video
                        continue
                    
                else:  # random segments
                    # Process with random segments
                    for j in range(gifs_per_video):
                        # Get video info to determine duration
                        try:
                            duration_cmd = [
                                FFPROBE_PATH,
                                '-v', 'error', 
                                '-show_entries', 'format=duration', 
                                '-of', 'default=noprint_wrappers=1:nokey=1', 
                                video_path
                            ]
                            
                            result = subprocess.run(duration_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                            if result.stdout.strip():
                                video_duration = float(result.stdout.strip())
                            else:
                                # If no duration found, use a default
                                video_duration = 10.0
                        except Exception:
                            video_duration = 10.0
                            
                        # Generate random segment
                        segment_length = random.uniform(min_length, max_length)
                        if segment_length > video_duration:
                            segment_length = video_duration
                            
                        max_start = video_duration - segment_length
                        if max_start <= 0:
                            # Video too short for specified segment length
                            continue
                            
                        start_time = random.uniform(0, max_start)
                        end_time = start_time + segment_length
                        
                        # Generate a new random seed
                        self.effect_seed.set(random.randint(1, 1000000))
                        
                        try:
                            # Create GIF with direct method
                            self._create_single_gif_direct(
                                start_time,
                                end_time,
                                playback_speed,
                                f"bulk_{i+1}_seg{j+1}_"
                            )
                        except Exception as e:
                            print(f"Error processing segment {j+1} of video {file_name}: {e}")
                            # Continue with next segment
                            continue
            
            # Restore original settings
            self.max_width_var.set(original_max_width)
            self.frame_skip_var.set(original_frame_skip)
            self.frame_duration_var.set(original_speed)
            self.quality_var.set(original_quality)
            
            # Update status
            self.status_var.set(f"Bulk conversion complete! Processed {len(video_files)} videos.")
            self.progress_var.set(100)
            
            # Show message box
            self.root.after(0, lambda: messagebox.showinfo("Bulk Conversion Complete", 
                            f"Successfully processed {len(video_files)} videos. GIFs saved to output folder."))
            
        except Exception as e:
            self.status_var.set(f"Error in bulk conversion: {str(e)}")
            print(f"Bulk conversion error: {str(e)}")
            
            # Show error message
            self.root.after(0, lambda: messagebox.showerror("Error", 
                          f"Error during bulk conversion: {str(e)}"))
            
        finally:
            self.is_processing = False
    
    def on_frame_configure(self, event):
        """Reset the scroll region to encompass the inner frame"""
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
    
    def on_canvas_configure(self, event):
        """When canvas is resized, resize the inner frame to match"""
        self.main_canvas.itemconfig(self.canvas_frame, width=event.width)
    
    def bind_mousewheel(self):
        """Bind mousewheel to scrolling for various platforms - robust approach"""
        def _on_mousewheel(event):
            # Cross-platform mouse wheel handling
            if platform.system() == 'Windows':
                self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            elif platform.system() == 'Darwin':  # macOS
                self.main_canvas.yview_scroll(int(-1*event.delta), "units")
            else:  # Linux
                if event.num == 4:
                    self.main_canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    self.main_canvas.yview_scroll(1, "units")
        
        # Unbind any existing mousewheel bindings to prevent duplicate bindings
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")
        
        # Rebind mousewheel for different platforms
        if platform.system() == 'Windows' or platform.system() == 'Darwin':
            self.root.bind_all("<MouseWheel>", _on_mousewheel)
        else:  # Linux
            self.root.bind_all("<Button-4>", _on_mousewheel)
            self.root.bind_all("<Button-5>", _on_mousewheel)
        
        # Ensure scrolling works when entering or leaving canvas
        def _bind_scroll(event):
            if platform.system() == 'Windows' or platform.system() == 'Darwin':
                self.root.bind_all("<MouseWheel>", _on_mousewheel)
            else:
                self.root.bind_all("<Button-4>", _on_mousewheel)
                self.root.bind_all("<Button-5>", _on_mousewheel)
        
        def _unbind_scroll(event):
            self.root.unbind_all("<MouseWheel>")
            self.root.unbind_all("<Button-4>")
            self.root.unbind_all("<Button-5>")
        
        # Bind to canvas and scrollable frame
        self.main_canvas.bind("<Enter>", _bind_scroll)
        self.main_canvas.bind("<Leave>", _unbind_scroll)
        
        # Ensure consistent scrolling for child widgets
        for widget in [self.main_canvas, self.scrollable_frame]:
            widget.bind("<Enter>", _bind_scroll)
            widget.bind("<Leave>", _unbind_scroll)
    
    def setup_main_tab(self):
        """Setup the main tab with file selection and preview"""
        # File selection frame
        file_frame = ttk.LabelFrame(self.main_frame, text="File Selection")
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(file_frame, text="Select Video/GIF", command=self.select_file).pack(side=tk.LEFT, padx=5, pady=5)
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        # Output mode selection frame
        output_mode_frame = ttk.LabelFrame(self.main_frame, text="Output Mode")
        output_mode_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Radiobutton(output_mode_frame, text="Video to GIF", 
                      variable=self.output_mode_var, value="gif").pack(side=tk.LEFT, padx=20, pady=5)
        ttk.Radiobutton(output_mode_frame, text="Video to Video", 
                      variable=self.output_mode_var, value="video").pack(side=tk.LEFT, padx=20, pady=5)
        
        # Video preview container (initially empty)
        self.preview_container = ttk.Frame(self.main_frame)
        self.preview_container.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)
        
        # Video controls frame (only visible for videos)
        self.video_controls_frame = ttk.LabelFrame(self.main_frame, text="Video Controls")
        
        # Time selection
        time_frame = ttk.Frame(self.video_controls_frame)
        time_frame.pack(fill=tk.X, pady=5)
        
        # Start Time Slider
        ttk.Label(time_frame, text="Start Time (sec):").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.start_scale = ttk.Scale(time_frame, 
                                     from_=0, 
                                     to=10,  # Will be updated dynamically with video duration 
                                     orient=tk.HORIZONTAL, 
                                     length=200, 
                                     variable=self.start_time_var,
                                     command=self.on_scale_change)
        self.start_scale.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.start_entry = ttk.Entry(time_frame, textvariable=self.start_time_var, width=8)
        self.start_entry.grid(row=0, column=2, padx=5, pady=2)
        
        # End Time Slider
        ttk.Label(time_frame, text="End Time (sec):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.end_scale = ttk.Scale(time_frame, 
                                   from_=0, 
                                   to=10,  # Will be updated dynamically with video duration
                                   orient=tk.HORIZONTAL, 
                                   length=200, 
                                   variable=self.end_time_var,
                                   command=self.on_scale_change)
        self.end_scale.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        self.end_entry = ttk.Entry(time_frame, textvariable=self.end_time_var, width=8)
        self.end_entry.grid(row=1, column=2, padx=5, pady=2)
        
        # Duration Label
        ttk.Label(time_frame, text="Duration (sec):").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.duration_label = ttk.Label(time_frame, text="0.0")
        self.duration_label.grid(row=2, column=1, padx=5, pady=2)
        
        time_frame.columnconfigure(2, weight=1)
        
        # Bind events to update duration and validate input
        self.start_entry.bind('<Return>', self.on_time_entry_change)
        self.start_entry.bind('<FocusOut>', self.on_time_entry_change)
        self.end_entry.bind('<Return>', self.on_time_entry_change)
        self.end_entry.bind('<FocusOut>', self.on_time_entry_change)
        
        # Frame extraction button
        extract_frame = ttk.Frame(self.video_controls_frame)
        extract_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(extract_frame, text="Extract Frames", command=self.extract_frames).pack(side=tk.LEFT, padx=5)
        ttk.Button(extract_frame, text="Jump to Start", command=lambda: self.jump_to_frame("start")).pack(side=tk.LEFT, padx=5)
        ttk.Button(extract_frame, text="Jump to End", command=lambda: self.jump_to_frame("end")).pack(side=tk.LEFT, padx=5)
        
        # Output options frame
        options_frame = ttk.LabelFrame(self.main_frame, text="Output Options")
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Common options
        common_frame = ttk.Frame(options_frame)
        common_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(common_frame, text="Max Width (px):").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(common_frame, from_=100, to=1200, increment=10, textvariable=self.max_width_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(common_frame, text="(Smaller = faster processing)").pack(side=tk.LEFT, padx=5)
        
        # Frame skip (common but more relevant for GIF)
        skip_frame = ttk.Frame(options_frame)
        skip_frame.pack(fill=tk.X, pady=2)
        ttk.Label(skip_frame, text="Frame Skip:").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(skip_frame, from_=1, to=10, textvariable=self.frame_skip_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(skip_frame, text="(Higher values = smaller file)").pack(side=tk.LEFT, padx=5)
        
        # Playback speed (common)
        duration_frame = ttk.Frame(options_frame)
        duration_frame.pack(fill=tk.X, pady=2)
        ttk.Label(duration_frame, text="Playback Speed:").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(duration_frame, from_=0.1, to=3.0, increment=0.1, textvariable=self.frame_duration_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(duration_frame, text="(1.0 = original speed, 2.0 = faster, 0.5 = slower)").pack(side=tk.LEFT, padx=5)
        
        # Consistent effects (common)
        consistency_frame = ttk.Frame(options_frame)
        consistency_frame.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(consistency_frame, text="Use consistent effects across all frames", 
                      variable=self.consistent_effects).pack(side=tk.LEFT, padx=5)
        ttk.Label(consistency_frame, text="Seed:").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(consistency_frame, from_=1, to=1000000, textvariable=self.effect_seed, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Button(consistency_frame, text="New Seed", 
                 command=lambda: self.effect_seed.set(random.randint(1, 1000000))).pack(side=tk.LEFT, padx=5)
        
        # GIF-specific options frame
        self.gif_options_frame = ttk.LabelFrame(options_frame, text="GIF-Specific Options")
        self.gif_options_frame.pack(fill=tk.X, pady=5)
        
        # Quality
        quality_frame = ttk.Frame(self.gif_options_frame)
        quality_frame.pack(fill=tk.X, pady=2)
        ttk.Label(quality_frame, text="Quality (1-100):").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(quality_frame, from_=1, to=100, textvariable=self.quality_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # GIF animation options
        anim_frame = ttk.Frame(self.gif_options_frame)
        anim_frame.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(anim_frame, text="Loop GIF", variable=self.loop_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(anim_frame, text="Reverse Frames", variable=self.reverse_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(anim_frame, text="Ping-Pong Effect", variable=self.ping_pong_var).pack(side=tk.LEFT, padx=5)
        
        # Video-specific options frame
        self.video_options_frame = ttk.LabelFrame(options_frame, text="Video-Specific Options")
        self.video_options_frame.pack(fill=tk.X, pady=5)
        
        # Video format
        format_frame = ttk.Frame(self.video_options_frame)
        format_frame.pack(fill=tk.X, pady=2)
        ttk.Label(format_frame, text="Format:").pack(side=tk.LEFT, padx=5)
        format_combo = ttk.Combobox(format_frame, textvariable=self.video_format_var, width=5, 
                                  values=["mp4", "avi", "mov", "mkv"])
        format_combo.pack(side=tk.LEFT, padx=5)
        
        # Video codec
        codec_frame = ttk.Frame(self.video_options_frame)
        codec_frame.pack(fill=tk.X, pady=2)
        ttk.Label(codec_frame, text="Codec:").pack(side=tk.LEFT, padx=5)
        codec_combo = ttk.Combobox(codec_frame, textvariable=self.video_codec_var, width=8, 
                                 values=["h264", "mpeg4", "libx264", "libxvid"])
        codec_combo.pack(side=tk.LEFT, padx=5)
        
        # Video bitrate
        bitrate_frame = ttk.Frame(self.video_options_frame)
        bitrate_frame.pack(fill=tk.X, pady=2)
        ttk.Label(bitrate_frame, text="Bitrate:").pack(side=tk.LEFT, padx=5)
        bitrate_combo = ttk.Combobox(bitrate_frame, textvariable=self.video_bitrate_var, width=6, 
                                   values=["500k", "1M", "2M", "5M", "10M"])
        bitrate_combo.pack(side=tk.LEFT, padx=5)
        
        # Update visible options based on selected mode
        self.output_mode_var.trace_add("write", self.update_output_options)
        
        # Multi-copy options frame
        multi_copy_frame = ttk.LabelFrame(self.main_frame, text="Multiple Copies")
        multi_copy_frame.pack(fill=tk.X, padx=10, pady=5)

        # Quantity selection
        quantity_frame = ttk.Frame(multi_copy_frame)
        quantity_frame.pack(fill=tk.X, pady=2)
        ttk.Label(quantity_frame, text="Number of Copies:").pack(side=tk.LEFT, padx=5)
        self.copies_var = tk.IntVar(value=1)
        ttk.Spinbox(quantity_frame, from_=1, to=50, textvariable=self.copies_var, width=5).pack(side=tk.LEFT, padx=5)

        # Copy type selection
        type_frame = ttk.Frame(multi_copy_frame)
        type_frame.pack(fill=tk.X, pady=2)
        self.copy_type_var = tk.StringVar(value="exact")
        ttk.Radiobutton(type_frame, text="Exact Copies (Same Segment)", 
                      variable=self.copy_type_var, value="exact").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(type_frame, text="Random Segments", 
                      variable=self.copy_type_var, value="random").pack(side=tk.LEFT, padx=5)

        # Settings for random segments
        random_settings_frame = ttk.Frame(multi_copy_frame)
        random_settings_frame.pack(fill=tk.X, pady=2)
        ttk.Label(random_settings_frame, text="Min Length (sec):").pack(side=tk.LEFT, padx=5)
        self.random_min_length_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(random_settings_frame, from_=1.0, to=10.0, increment=0.5, 
                  textvariable=self.random_min_length_var, width=5).pack(side=tk.LEFT, padx=5)

        ttk.Label(random_settings_frame, text="Max Length (sec):").pack(side=tk.LEFT, padx=5)
        self.random_max_length_var = tk.DoubleVar(value=3.0)
        ttk.Spinbox(random_settings_frame, from_=1.0, to=30.0, increment=0.5,
                  textvariable=self.random_max_length_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # Processed frames preview (shown after extraction)
        self.frames_frame = ttk.LabelFrame(self.main_frame, text="Frame Preview")
        
        # Add a button to go back to video preview
        back_btn_frame = ttk.Frame(self.frames_frame)
        back_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(back_btn_frame, text="Back to Video", command=self.show_video_preview).pack(side=tk.LEFT, padx=5)
        
        self.preview_label = ttk.Label(self.frames_frame)
        self.preview_label.pack(padx=5, pady=5)
        
        # Frame navigation
        nav_frame = ttk.Frame(self.frames_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(nav_frame, text="Previous Frame", command=self.prev_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next Frame", command=self.next_frame).pack(side=tk.LEFT, padx=5)
        self.frame_label = ttk.Label(nav_frame, text="Frame: 0/0")
        self.frame_label.pack(side=tk.LEFT, padx=5)
        
        # Process buttons
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Preview Effect", command=self.preview_effect).pack(side=tk.LEFT, padx=5)
        self.generate_button = ttk.Button(button_frame, text="Generate GIF/Video", command=self.start_processing)
        self.generate_button.pack(side=tk.LEFT, padx=5)
        self.bulk_button = ttk.Button(button_frame, text="Bulk Generate GIFs/Videos", command=self.start_processing)
        self.bulk_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Open Output Folder", command=self.open_output_folder).pack(side=tk.LEFT, padx=5)
    
        ttk.Button(button_frame, text="Bulk Export Frames", 
         command=self.bulk_export_frames_as_images).pack(side=tk.LEFT, padx=5)
         
        # Initialize options visibility based on default mode
        self.update_output_options()
 
    def show_video_preview(self):
        """Show the video preview and hide the frames preview"""
        # Hide frames preview if visible
        if hasattr(self, 'frames_frame'):
            self.frames_frame.pack_forget()
    
        # Show video preview if it exists
        if hasattr(self, 'video_preview') and self.video_preview:
            # Make sure it's packed in the preview container
            self.preview_container.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)
            self.video_preview.pack(fill=tk.BOTH, expand=True)
        
            # Update status
            self.status_var.set("Video preview restored")
        else:
            self.status_var.set("No video preview available")
    
    def update_output_options(self, *args):
        """Update the visible options based on the selected output mode"""
        selected_mode = self.output_mode_var.get()
        
        if selected_mode == "gif":
            # Show GIF options, hide Video options
            self.gif_options_frame.pack(fill=tk.X, pady=5)
            self.video_options_frame.pack_forget()
            self.generate_button.config(text="Generate GIF/Video")
            self.bulk_button.config(text="Bulk Generate GIFs/Videos")
        else:  # video
            # Show Video options, hide GIF options
            self.gif_options_frame.pack_forget()
            self.video_options_frame.pack(fill=tk.X, pady=5)
            self.generate_button.config(text="Generate GIF/Video")
            self.bulk_button.config(text="Bulk Generate GIFs/Videos")
    
    def on_time_entry_change(self, event=None):
        """Handle time entry changes"""
        try:
            # Convert entries to floats
            start_time = float(self.start_entry.get())
            end_time = float(self.end_entry.get())
            
            # Ensure start time is not negative and not greater than end time
            start_time = max(0, start_time)
            end_time = max(start_time, end_time)
            
            # Update variables
            self.start_time_var.set(start_time)
            self.end_time_var.set(end_time)
            
            # Update duration
            self.update_duration()
            
            # Update video preview selection if available
            if self.video_preview and hasattr(self.video_preview, 'update_selection'):
                self.video_preview.update_selection(start_time, end_time)
        
        except ValueError:
            # Reset to previous valid values if parsing fails
            self.start_entry.delete(0, tk.END)
            self.start_entry.insert(0, f"{self.start_time_var.get():.1f}")
            self.end_entry.delete(0, tk.END)
            self.end_entry.insert(0, f"{self.end_time_var.get():.1f}")

    def extract_frames(self):
        """Extract frames from video between start and end times"""
        if not self.is_video or not self.input_path:
            return
     
        # Reset frames
        self.frames = []
    
        # Get start and end times
        start_time = self.start_time_var.get()
        end_time = self.end_time_var.get()
        duration = end_time - start_time
    
        # Update status
        self.status_var.set(f"Extracting frames from {start_time:.1f}s to {end_time:.1f}s...")
        self.progress_var.set(0)
        self.root.update()
    
        # Start extraction in a separate thread
        threading.Thread(target=self.extract_frames_thread, 
                       args=(start_time, end_time, duration), 
                       daemon=True).start()

    def update_selection_from_timeline(self, start_time, end_time):
        """Update time selection from timeline"""
        # Update entry fields
        self.start_entry.delete(0, tk.END)
        self.start_entry.insert(0, f"{start_time:.1f}")
        self.end_entry.delete(0, tk.END)
        self.end_entry.insert(0, f"{end_time:.1f}")
        
        # Update variables
        self.start_time_var.set(start_time)
        self.end_time_var.set(end_time)
        
        # Update duration
        self.update_duration()
    
    def get_video_info(self):
        """Get video duration and other info using FFmpeg"""
        try:
            # Use FFprobe to get video duration
            duration_cmd = [
                FFPROBE_PATH,
                '-v', 'error', 
                '-show_entries', 'format=duration', 
                '-of', 'default=noprint_wrappers=1:nokey=1', 
                self.input_path
            ]
            
            result = subprocess.run(duration_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Parse duration
            try:
                duration = float(result.stdout.strip())
            except ValueError:
                # Fallback if duration parsing fails
                duration = 10.0  # Default duration
            
            # Get frame rate
            fps_cmd = [
                FFPROBE_PATH,
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                self.input_path
            ]
            
            fps_result = subprocess.run(fps_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            fps_str = fps_result.stdout.strip()
            
            # Convert frame rate (often returned as fraction like "30000/1001")
            try:
                if '/' in fps_str:
                    num, den = fps_str.split('/')
                    frame_rate = round(float(num) / float(den))
                else:
                    frame_rate = round(float(fps_str))
            except (ValueError, ZeroDivisionError):
                # Fallback if frame rate parsing fails
                frame_rate = 30  # Default frame rate
            
            # Use after method to update UI safely
            def update_ui():
                try:
                    # Update video duration
                    self.video_duration = duration
                    self.frame_rate = frame_rate
                    
                    # Configure scales with actual video duration
                    if hasattr(self, 'start_scale'):
                        self.start_scale.config(from_=0, to=duration)
                    if hasattr(self, 'end_scale'):
                        self.end_scale.config(from_=0, to=duration)
                    
                    # Set start and end time - CHANGED HERE
                    self.start_time_var.set(0.0)
                    # Always set end time to full video duration
                    self.end_time_var.set(duration)
                    
                    # Update duration display
                    self.update_duration()
                    
                    # Update status
                    self.status_var.set(f"Video loaded: {duration:.1f} seconds, {frame_rate} fps")
                
                except Exception as ui_error:
                    print(f"Error updating video info UI: {ui_error}")
                    self.status_var.set(f"Partial error loading video: {ui_error}")
            
            # Schedule UI update on main thread
            self.root.after(0, update_ui)
        
        except Exception as e:
            # Use after to handle errors safely
            def show_error():
                self.status_var.set(f"Error getting video info: {str(e)}")
                print(f"Video info error details: {str(e)}")
            
            # Schedule error display on main thread
            self.root.after(0, show_error)
    
    def jump_to_frame(self, position):
        """Jump to start or end frame in video preview"""
        if not self.is_video:
            return
        
        if position == "start":
            time_pos = self.start_time_var.get()
        else:  # end
            time_pos = self.end_time_var.get()
        
        # Update video preview position if available
        if hasattr(self, 'video_preview') and self.video_preview:
            # Calculate frame position
            if hasattr(self.video_preview, 'frames') and self.video_preview.frames:
                frame_idx = int((time_pos / self.video_preview.video_duration) * len(self.video_preview.frames))
                frame_idx = max(0, min(frame_idx, len(self.video_preview.frames) - 1))
                
                # Update frame display
                self.video_preview.current_frame = frame_idx
                self.video_preview.preview_label.config(image=self.video_preview.frames[frame_idx])
                self.video_preview.progress_var.set(frame_idx)
                
                # Update time display
                mins = int(time_pos // 60)
                secs = time_pos % 60
                self.video_preview.time_display.config(text=f"{mins:02d}:{secs:06.3f}")
        
        # If frames are extracted, also jump to the right position there
        if self.frames and self.frames_frame.winfo_ismapped():
            # Extract a frame at the specified position
            try:
                frame_path = os.path.join(self.temp_dir, f"{position}_frame.jpg")
                cmd = [
                    FFMPEG_PATH,
                    '-i', self.input_path,
                    '-ss', str(time_pos),  # position in seconds
                    '-vframes', '1',  # extract 1 frame
                    '-q:v', '2',  # high quality
                    frame_path
                ]
                
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Display the frame if it exists
                if os.path.exists(frame_path):
                    img = Image.open(frame_path)
                    self.display_frame(img)
                    self.status_var.set(f"Jumped to {position} position at {time_pos:.1f} seconds")
            
            except Exception as e:
                self.status_var.set(f"Error jumping to frame: {str(e)}")
                print(f"Error details: {str(e)}")
    
    def setup_effects_tab(self):
        """Setup the effects tab with all effect parameters"""
        # Scrollable frame for effects
        canvas = tk.Canvas(self.effects_frame)
        scrollbar = ttk.Scrollbar(self.effects_frame, orient="vertical", command=canvas.yview)
        
        # Configure the canvas
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        # Create a frame inside the canvas
        effects_content = ttk.Frame(canvas)
        
        # Add the frame to the canvas
        canvas.create_window((0, 0), window=effects_content, anchor="nw")
        
        # Pack the canvas and scrollbar
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind mousewheel to scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Brightness
        brightness_frame = ttk.LabelFrame(effects_content, text="Brightness")
        brightness_frame.pack(fill=tk.X, pady=5, padx=10)
        
        ttk.Checkbutton(brightness_frame, text="Enable", variable=self.brightness_enabled).pack(side=tk.LEFT, padx=5)
        ttk.Label(brightness_frame, text="Min:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(brightness_frame, textvariable=self.brightness_min_var, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(brightness_frame, text="Max:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(brightness_frame, textvariable=self.brightness_max_var, width=6).pack(side=tk.LEFT, padx=5)
        
        # Contrast
        contrast_frame = ttk.LabelFrame(effects_content, text="Contrast")
        contrast_frame.pack(fill=tk.X, pady=5, padx=10)
        
        ttk.Checkbutton(contrast_frame, text="Enable", variable=self.contrast_enabled).pack(side=tk.LEFT, padx=5)
        ttk.Label(contrast_frame, text="Min:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(contrast_frame, textvariable=self.contrast_min_var, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(contrast_frame, text="Max:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(contrast_frame, textvariable=self.contrast_max_var, width=6).pack(side=tk.LEFT, padx=5)
        
        # Saturation
        saturation_frame = ttk.LabelFrame(effects_content, text="Saturation")
        saturation_frame.pack(fill=tk.X, pady=5, padx=10)
        
        ttk.Checkbutton(saturation_frame, text="Enable", variable=self.saturation_enabled).pack(side=tk.LEFT, padx=5)
        ttk.Label(saturation_frame, text="Min:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(saturation_frame, textvariable=self.saturation_min_var, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(saturation_frame, text="Max:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(saturation_frame, textvariable=self.saturation_max_var, width=6).pack(side=tk.LEFT, padx=5)
        
        # Blur
        blur_frame = ttk.LabelFrame(effects_content, text="Blur")
        blur_frame.pack(fill=tk.X, pady=5, padx=10)
        
        ttk.Checkbutton(blur_frame, text="Enable", variable=self.blur_enabled).pack(side=tk.LEFT, padx=5)
        ttk.Label(blur_frame, text="Min:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(blur_frame, textvariable=self.blur_min_var, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(blur_frame, text="Max:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(blur_frame, textvariable=self.blur_max_var, width=6).pack(side=tk.LEFT, padx=5)
        
        # Rotation
        rotation_frame = ttk.LabelFrame(effects_content, text="Rotation")
        rotation_frame.pack(fill=tk.X, pady=5, padx=10)
        
        ttk.Checkbutton(rotation_frame, text="Enable", variable=self.rotation_enabled).pack(side=tk.LEFT, padx=5)
        ttk.Label(rotation_frame, text="Min:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(rotation_frame, textvariable=self.rotation_min_var, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(rotation_frame, text="Max:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(rotation_frame, textvariable=self.rotation_max_var, width=6).pack(side=tk.LEFT, padx=5)
        
        # Resize
        resize_frame = ttk.LabelFrame(effects_content, text="Resize")
        resize_frame.pack(fill=tk.X, pady=5, padx=10)
        
        ttk.Checkbutton(resize_frame, text="Enable", variable=self.resize_enabled).pack(side=tk.LEFT, padx=5)
        ttk.Label(resize_frame, text="Min:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(resize_frame, textvariable=self.resize_min_var, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(resize_frame, text="Max:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(resize_frame, textvariable=self.resize_max_var, width=6).pack(side=tk.LEFT, padx=5)
        
        # Hue shift
        hue_frame = ttk.LabelFrame(effects_content, text="Hue Shift")
        hue_frame.pack(fill=tk.X, pady=5, padx=10)
    
        self.hue_shift_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(hue_frame, text="Enable", variable=self.hue_shift_enabled).pack(side=tk.LEFT, padx=5)
        ttk.Label(hue_frame, text="Min:").pack(side=tk.LEFT, padx=5)
        self.hue_min_var = tk.DoubleVar(value=-5)
        ttk.Entry(hue_frame, textvariable=self.hue_min_var, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(hue_frame, text="Max:").pack(side=tk.LEFT, padx=5)
        self.hue_max_var = tk.DoubleVar(value=5)
        ttk.Entry(hue_frame, textvariable=self.hue_max_var, width=6).pack(side=tk.LEFT, padx=5)
    
        # Subtle noise
        noise_frame = ttk.LabelFrame(effects_content, text="Subtle Noise")
        noise_frame.pack(fill=tk.X, pady=5, padx=10)
    
        self.noise_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(noise_frame, text="Enable", variable=self.noise_enabled).pack(side=tk.LEFT, padx=5)
        ttk.Label(noise_frame, text="Level:").pack(side=tk.LEFT, padx=5)
        self.noise_level_var = tk.DoubleVar(value=1.5)
        ttk.Scale(noise_frame, from_=0, to=5, variable=self.noise_level_var, 
                  orient=tk.HORIZONTAL).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
        # Region blending option
        region_frame = ttk.LabelFrame(effects_content, text="Region Effects")
        region_frame.pack(fill=tk.X, pady=5, padx=10)

        self.region_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(region_frame, text="Enable", variable=self.region_enabled).pack(side=tk.LEFT, padx=5)

        ttk.Label(region_frame, text="Blend:").pack(side=tk.LEFT, padx=5)
        self.region_blend_var = tk.IntVar(value=50)
        ttk.Scale(region_frame, from_=0, to=100, variable=self.region_blend_var, 
              orient=tk.HORIZONTAL).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Preset buttons (this section likely already exists)
        preset_frame = ttk.Frame(effects_content)
        preset_frame.pack(fill=tk.X, pady=10, padx=10)
        
        ttk.Button(preset_frame, text="Normal Wash", command=self.set_normal_wash).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="Deep Wash", command=self.set_deep_wash).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="Extreme Wash", command=self.set_extreme_wash).pack(side=tk.LEFT, padx=5)
    
    def setup_auto_tab(self):
        """Setup the auto generation tab"""
        # Auto generation options
        auto_frame = ttk.LabelFrame(self.auto_frame, text="Automatic GIF Generation")
        auto_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Enable auto generation
        enable_frame = ttk.Frame(auto_frame)
        enable_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(enable_frame, text="Enable Automatic GIF Generation", 
                      variable=self.auto_gen_enabled).pack(side=tk.LEFT, padx=5)
        
        # Number of GIFs
        num_frame = ttk.Frame(auto_frame)
        num_frame.pack(fill=tk.X, pady=5)
        ttk.Label(num_frame, text="Number of GIFs to Generate:").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(num_frame, from_=1, to=50, textvariable=self.num_gifs_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # GIF length range
        length_frame = ttk.Frame(auto_frame)
        length_frame.pack(fill=tk.X, pady=5)
        ttk.Label(length_frame, text="Minimum GIF Length (sec):").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(length_frame, from_=0.5, to=30, increment=0.5, 
                  textvariable=self.min_gif_length_var, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(length_frame, text="Maximum GIF Length (sec):").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(length_frame, from_=1, to=60, increment=0.5,
                  textvariable=self.max_gif_length_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # Description
        desc_frame = ttk.LabelFrame(self.auto_frame, text="Description")
        desc_frame.pack(fill=tk.BOTH, padx=10, pady=10, expand=True)
        
        description = """
        Automatic GIF Generation will:
        
        1. Randomly select segments from the video
        2. Generate unique GIFs with different effects for each segment
        3. Save all GIFs to the output folder
        
        This is useful for creating a variety of GIFs from a single video.
        Just enable this option and press "Generate GIF" on the main tab.
        """
        
        ttk.Label(desc_frame, text=description, wraplength=500, justify="left").pack(padx=10, pady=10)
    
    def set_normal_wash(self):
        """Set parameters for normal wash"""
        self.brightness_min_var.set(0.8)
        self.brightness_max_var.set(1.2)
        self.contrast_min_var.set(0.8)
        self.contrast_max_var.set(1.2)
        self.saturation_min_var.set(0.8)
        self.saturation_max_var.set(1.2)
        self.blur_min_var.set(0)
        self.blur_max_var.set(0.5)
        self.rotation_min_var.set(-2)
        self.rotation_max_var.set(2)
        self.resize_min_var.set(0.95)
        self.resize_max_var.set(1.05)
        if hasattr(self, 'hue_min_var'):
            self.hue_min_var.set(-5)  # For normal wash
            self.hue_max_var.set(5)   # For normal wash
        if hasattr(self, 'noise_level_var'):
            self.noise_level_var.set(1.0)  # For normal wash
    
    def set_deep_wash(self):
        """Set parameters for deep wash"""
        self.brightness_min_var.set(0.7)
        self.brightness_max_var.set(1.3)
        self.contrast_min_var.set(0.7)
        self.contrast_max_var.set(1.3)
        self.saturation_min_var.set(0.7)
        self.saturation_max_var.set(1.3)
        self.blur_min_var.set(0)
        self.blur_max_var.set(1.0)
        self.rotation_min_var.set(-3)
        self.rotation_max_var.set(3)
        self.resize_min_var.set(0.9)
        self.resize_max_var.set(1.1)
        if hasattr(self, 'hue_min_var'):
            self.hue_min_var.set(-8)  # For deep wash (more color variation)
            self.hue_max_var.set(8)   # For deep wash
        if hasattr(self, 'noise_level_var'):
            self.noise_level_var.set(2.0)  # For deep wash (double the noise)
    
    def set_extreme_wash(self):
        """Set parameters for extreme wash"""
        self.brightness_min_var.set(0.6)
        self.brightness_max_var.set(1.4)
        self.contrast_min_var.set(0.6)
        self.contrast_max_var.set(1.4)
        self.saturation_min_var.set(0.6)
        self.saturation_max_var.set(1.4)
        self.blur_min_var.set(0)
        self.blur_max_var.set(1.5)
        self.rotation_min_var.set(-5)
        self.rotation_max_var.set(5)
        self.resize_min_var.set(0.85)
        self.resize_max_var.set(1.15)
        if hasattr(self, 'hue_min_var'):
            self.hue_min_var.set(-12)  # For extreme wash (significant color variation)
            self.hue_max_var.set(12)   # For extreme wash
        if hasattr(self, 'noise_level_var'):
            self.noise_level_var.set(3.5)  # For extreme wash (much stronger noise)
    
    def select_file(self):
        """Open file dialog to select a file or folder"""
        # Create custom dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Type")
        dialog.geometry("300x120")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() - dialog.winfo_width()) // 2
        y = (dialog.winfo_screenheight() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
        
        # Dialog content
        tk.Label(dialog, text="Are you picking a video or folder?", 
                 pady=15).pack()
        
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=10)
        
        result = [None]  # Use list to store result
        
        def on_video():
                result[0] = "Video"
                dialog.destroy()
                
        def on_folder():
                result[0] = "Folder"
                dialog.destroy()
        
        tk.Button(button_frame, text="Video", width=8, command=on_video).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Folder", width=8, command=on_folder).pack(side=tk.LEFT, padx=10)
        
        # Wait for dialog
        self.root.wait_window(dialog)
        
        # Process selection based on result
        if result[0] == "Folder":
                # Folder selection code
                folder_path = filedialog.askdirectory(
                        title="Select Folder Containing Videos"
                )
                
                if folder_path:
                        # Store folder path and set folder mode
                        self.input_folder = folder_path
                        self.is_folder_mode = True
                        
                        # Count video files in the folder
                        video_files = self.count_video_files(folder_path)
                        
                        # Update UI
                        self.file_label.config(text=f"Selected folder: {os.path.basename(folder_path)} ({video_files} videos)")
                        self.status_var.set(f"Ready to process {video_files} videos from folder")
                        
                        # Hide video controls and preview
                        if hasattr(self, 'video_controls_frame'):
                                self.video_controls_frame.pack_forget()
                        
                        if hasattr(self, 'video_preview') and self.video_preview:
                                self.video_preview.pack_forget()
                
        elif result[0] == "Video":
                # File selection code
                file_path = filedialog.askopenfilename(
                        title="Select Video or GIF File",
                        filetypes=[
                                ("Video/GIF files", "*.mp4 *.avi *.mov *.wmv *.mkv *.flv *.gif"),
                                ("All files", "*.*")
                        ]
                )
                
                if file_path:
                        # Reset folder mode
                        self.is_folder_mode = False
                        
                        # Update file path and label on the main thread
                        self.input_path = file_path
                        self.file_label.config(text=os.path.basename(file_path))
                    
                        # Determine if it's a video or GIF
                        file_ext = os.path.splitext(file_path)[1].lower()
                    
                        if file_ext == '.gif':
                                # GIF processing
                                self.is_video = False
                            
                                # Hide video controls if present
                                if hasattr(self, 'video_controls_frame'):
                                        self.video_controls_frame.pack_forget()
                            
                                # Update status
                                self.status_var.set("Loading GIF frames...")
                            
                                # Load GIF frames
                                try:
                                        self.load_gif_frames()
                                except Exception as e:
                                        self.status_var.set(f"Error loading GIF: {str(e)}")
                    
                        else:
                                # Video processing
                                self.is_video = True
                            
                                # Create or update video preview
                                if not hasattr(self, 'video_preview') or not self.video_preview:
                                        self.video_preview = VideoPreviewFrame(self.preview_container, file_path, FFMPEG_PATH)
                                        self.video_preview.pack(fill=tk.BOTH, expand=True)
                                else:
                                        # Update existing video preview
                                        self.video_preview.video_path = file_path
                                        self.video_preview.pack(fill=tk.BOTH, expand=True)
                                        self.video_preview.extract_frames()
                            
                                # Show video controls
                                if hasattr(self, 'video_controls_frame'):
                                        self.video_controls_frame.pack(fill=tk.X, padx=10, pady=5, after=self.preview_container)
                            
                                # Update status and get video info
                                self.status_var.set("Getting video info...")
                            
                                # Get video information
                                try:
                                        self.get_video_info()
                                except Exception as e:
                                        self.status_var.set(f"Error getting video info: {str(e)}")
    
    def count_video_files(self, folder_path):
        """Count the number of video files in a folder"""
        video_extensions = ('.mp4', '.avi', '.mov', '.wmv', '.mkv', '.flv')
        count = 0
        
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                count += 1
        
        return count
    
    
    def extract_frames_thread(self, start_time, end_time, duration):
        """Thread function to extract frames from video"""
        try:
            # Save current scroll position
            current_scroll = self.main_canvas.yview()[0] if hasattr(self, 'main_canvas') else 0
            
            # Clear temp directory of old frames
            for file in os.listdir(self.temp_dir):
                if file.startswith("frame_") and file.endswith(".jpg"):
                    os.remove(os.path.join(self.temp_dir, file))
            
            # Use FFmpeg to extract frames
            frame_skip = self.frame_skip_var.get()
            frame_rate = self.frame_rate // frame_skip
            max_width = self.max_width_var.get()
            
            # Calculate actual frames to extract
            cmd = [
                FFMPEG_PATH,
                '-i', self.input_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-vf', f'fps={frame_rate},scale=\'min({max_width},iw)\':-1',
                '-q:v', '1',
                os.path.join(self.temp_dir, 'frame_%04d.jpg')
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for process to complete
            process.wait()
            
            # Load extracted frames
            frame_files = sorted([f for f in os.listdir(self.temp_dir) 
                               if f.startswith("frame_") and f.endswith(".jpg")])
            
            self.frames = []
            total_frames = len(frame_files)
            
            for i, frame_file in enumerate(frame_files):
                img = Image.open(os.path.join(self.temp_dir, frame_file))
                self.frames.append(img)
                
                # Update progress
                progress = ((i + 1) / total_frames) * 100
                self.progress_var.set(progress)
                
                if i % 5 == 0:  # Update status every 5 frames
                    self.status_var.set(f"Loading frame {i+1}/{total_frames} ({progress:.1f}%)")
                    self.root.update_idletasks()
            
            # Update display
            self.current_frame_index = 0
            if self.frames:
                self.status_var.set(f"Loaded {len(self.frames)} frames")
                self.frame_label.config(text=f"Frame: 1/{len(self.frames)}")
                
                # Keep video preview in place, just add frame preview section
                # Don't hide video preview
                # if self.video_preview:
                #     self.video_preview.pack_forget()
                
                # Instead, update the frames_frame without moving video
                self.frames_frame.pack(fill=tk.BOTH, padx=10, pady=5, expand=True, after=self.video_controls_frame)
                self.display_frame(self.frames[0])
                
                # Restore scroll position
                if hasattr(self, 'main_canvas'):
                    self.main_canvas.yview_moveto(current_scroll)
            else:
                self.status_var.set("No frames extracted")
            
        except Exception as e:
            self.status_var.set(f"Error extracting frames: {str(e)}")
            print(f"Error details: {str(e)}")
    
    def _bulk_export_frames_thread(self, folder_path):
        """Thread function to export frames from all videos"""
        try:
            self.is_processing = True
            
            # Get all video files in folder
            video_files = self.get_video_files(folder_path)
            if not video_files:
                self.status_var.set("No video files found in selected folder")
                self.is_processing = False
                return
            
            # Create a timestamp for the main export
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            main_export_folder = os.path.join(self.output_folder, f"bulk_frames_{timestamp}")
            if not os.path.exists(main_export_folder):
                os.makedirs(main_export_folder)
                
            # Update status
            self.status_var.set(f"Processing {len(video_files)} videos for frame export")
            self.progress_var.set(0)
            self.root.update_idletasks()
            
            # Process each video
            for video_index, video_path in enumerate(video_files):
                # Update status for current video
                filename = os.path.basename(video_path)
                video_name = os.path.splitext(filename)[0]
                
                self.status_var.set(f"Extracting frames from video {video_index+1}/{len(video_files)}: {filename}")
                self.progress_var.set((video_index / len(video_files)) * 100)
                self.root.update_idletasks()
                
                # Create a folder for this video
                video_folder = os.path.join(main_export_folder, video_name)
                if not os.path.exists(video_folder):
                    os.makedirs(video_folder)
                
                # Extract frames
                try:
                    # Use FFmpeg to extract frames with high quality
                    max_width = self.max_width_var.get()
                    frame_skip = self.frame_skip_var.get()
                    
                    # Calculate target frame rate based on frame skip
                    # Assuming source is 30fps, reduce by frame skip factor
                    target_fps = 30 / frame_skip
                    
                    # Extract frames command
                    extract_cmd = [
                        FFMPEG_PATH,
                        '-i', video_path,
                        '-vf', f'fps={target_fps},scale={max_width}:-1',  # Simple scaling
                        '-q:v', '2',  # High quality
                        os.path.join(video_folder, f'frame_%04d.jpg')
                    ]
                    
                    # Run extraction
                    subprocess.run(extract_cmd, capture_output=True)
                    
                    # Count extracted frames
                    frame_count = len([f for f in os.listdir(video_folder) if f.endswith('.jpg')])
                    
                    # Update sub-progress
                    self.status_var.set(f"Extracted {frame_count} frames from {filename}")
                    self.root.update_idletasks()
                    
                except Exception as e:
                    print(f"Error extracting frames from {filename}: {e}")
                    # Continue with next video
            
            # Processing complete
            self.status_var.set(f"Bulk frame export complete! Processed {len(video_files)} videos.")
            self.progress_var.set(100)
            
            # Show message
            self.root.after(0, lambda: messagebox.showinfo("Bulk Export Complete", 
                          f"Successfully exported frames from {len(video_files)} videos to:\n{main_export_folder}"))
            
            # Open the output folder
            if os.path.exists(main_export_folder):
                # Open folder based on platform
                if sys.platform == 'win32':
                    os.startfile(main_export_folder)
                elif sys.platform == 'darwin':  # macOS
                    os.system(f'open "{main_export_folder}"')
                else:  # Linux
                    os.system(f'xdg-open "{main_export_folder}"')
                    
        except Exception as e:
            self.status_var.set(f"Error in bulk frame export: {str(e)}")
            print(f"Error in bulk frame export: {str(e)}")
            
        finally:
            self.is_processing = False
    
    
    def load_gif_frames(self):
        """Load frames from a GIF file"""
        try:
            self.frames = []
            
            # Open the GIF
            with Image.open(self.input_path) as img:
                # Get total frame count for progress
                frame_count = 0
                for _ in ImageSequence.Iterator(img):
                    frame_count += 1
                
                # Reset position
                img.seek(0)
                
                # Get frames with frame skip
                frame_skip = self.frame_skip_var.get()
                
                for i, frame in enumerate(ImageSequence.Iterator(img)):
                    if i % frame_skip == 0:
                        # Convert to RGB (removes transparency)
                        frame_copy = frame.convert('RGB')
                        self.frames.append(frame_copy)
                    
                    # Update progress
                    progress = ((i + 1) / frame_count) * 100
                    self.progress_var.set(progress)
                    
                    if i % 5 == 0:  # Update status every 5 frames
                        self.status_var.set(f"Loading frame {i+1}/{frame_count} ({progress:.1f}%)")
                        self.root.update_idletasks()
            
            # Update display
            self.current_frame_index = 0
            if self.frames:
                self.status_var.set(f"Loaded {len(self.frames)} frames from GIF")
                self.frame_label.config(text=f"Frame: 1/{len(self.frames)}")
                
                # Replace preview container with frames view
                self.preview_container.pack_forget()
                self.frames_frame.pack(fill=tk.BOTH, padx=10, pady=5, expand=True, before=self.video_controls_frame)
                self.display_frame(self.frames[0])
            else:
                self.status_var.set("No frames loaded from GIF")
            
        except Exception as e:
            self.status_var.set(f"Error loading GIF: {str(e)}")
            print(f"Error details: {str(e)}")
    
    def display_frame(self, frame):
        """Display a frame in the preview label"""
        try:
            # Make a copy to avoid modifying original
            display_img = frame.copy()
            
            # Get max dimensions for display
            max_width = 400
            max_height = 300
            
            # Calculate new dimensions while maintaining aspect ratio
            width, height = display_img.size
            aspect_ratio = width / height
            
            if width > height:
                new_width = min(width, max_width)
                new_height = int(new_width / aspect_ratio)
                if new_height > max_height:
                    new_height = max_height
                    new_width = int(new_height * aspect_ratio)
            else:
                new_height = min(height, max_height)
                new_width = int(new_height * aspect_ratio)
                if new_width > max_width:
                    new_width = max_width
                    new_height = int(new_width / aspect_ratio)
            
            # Resize image for display
            display_img = display_img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(display_img)
            
            # Update label
            self.preview_label.config(image=photo)
            self.preview_label.image = photo  # Keep a reference
            
        except Exception as e:
            print(f"Error displaying frame: {str(e)}")
    
    def next_frame(self):
        """Show next frame in preview"""
        if not self.frames:
            return
        
        self.current_frame_index = (self.current_frame_index + 1) % len(self.frames)
        self.display_frame(self.frames[self.current_frame_index])
        self.frame_label.config(text=f"Frame: {self.current_frame_index + 1}/{len(self.frames)}")
    
    def prev_frame(self):
        """Show previous frame in preview"""
        if not self.frames:
            return
        
        self.current_frame_index = (self.current_frame_index - 1) % len(self.frames)
        self.display_frame(self.frames[self.current_frame_index])
        self.frame_label.config(text=f"Frame: {self.current_frame_index + 1}/{len(self.frames)}")
    
    def preview_effect(self):
        """Preview effect on current frame"""
        if not self.frames or self.current_frame_index >= len(self.frames):
            self.status_var.set("No frame to preview")
            return
        
        # Get current frame
        frame = self.frames[self.current_frame_index].copy()
        
        # Apply effects
        processed_frame = self.apply_effects_to_frame(frame, self.current_frame_index)
        
        # Display processed frame
        self.display_frame(processed_frame)
        self.status_var.set("Previewing effects on current frame")
    
    def apply_consistent_effects(self):
        """Generate consistent effect parameters for all frames"""
        # Set fixed random seed for consistency
        random.seed(self.effect_seed.get())
        
        # Generate a single set of effect parameters for all frames
        self.effect_params = {
            'brightness': random.uniform(
                self.brightness_min_var.get(),
                self.brightness_max_var.get()
            ) if self.brightness_enabled.get() else 1.0,
            
            'contrast': random.uniform(
                self.contrast_min_var.get(),
                self.contrast_max_var.get()
            ) if self.contrast_enabled.get() else 1.0,
            
            'saturation': random.uniform(
                self.saturation_min_var.get(),
                self.saturation_max_var.get()
            ) if self.saturation_enabled.get() else 1.0,
            
            'blur': random.uniform(
                self.blur_min_var.get(),
                self.blur_max_var.get()
            ) if self.blur_enabled.get() else 0,
            
            'rotation': random.uniform(
                self.rotation_min_var.get(),
                self.rotation_max_var.get()
            ) if self.rotation_enabled.get() else 0,
            
            'resize': random.uniform(
                self.resize_min_var.get(),
                self.resize_max_var.get()
            ) if self.resize_enabled.get() else 1.0
        }
        
        # Add hue shift if enabled
        if hasattr(self, 'hue_shift_enabled') and self.hue_shift_enabled.get():
            self.effect_params['hue_shift'] = random.uniform(
                self.hue_min_var.get(),
                self.hue_max_var.get()
            )
        
        # Add noise if enabled
        if hasattr(self, 'noise_enabled') and self.noise_enabled.get():
            self.effect_params['noise'] = self.noise_level_var.get()
        
        # Add subtle variations that won't be visually noticeable but will make each GIF unique
        self.unique_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=16))
        self.invisible_params = {
            'quality': random.randint(97, 99),
            'unique_hash': self.unique_id
        }
    
    def apply_effects_to_frame(self, frame, frame_index=0):
        """Apply effects without corrupting the image"""
        try:
            # Make a copy of the frame
            img = frame.copy()
            
            # Generate parameters if needed
            if not hasattr(self, 'effect_params') or not self.consistent_effects.get():
                self.apply_consistent_effects()
            
            # Save original dimensions
            original_dimensions = img.size
            
            # Apply safe effects - ONLY safe color adjustments
            
            # Brightness (safe)
            if self.brightness_enabled.get():
                brightness = min(1.5, max(0.5, self.effect_params['brightness']))  # Limit range
                img = ImageEnhance.Brightness(img).enhance(brightness)
            
            # Contrast (safe)
            if self.contrast_enabled.get():
                contrast = min(1.5, max(0.5, self.effect_params['contrast']))  # Limit range
                img = ImageEnhance.Contrast(img).enhance(contrast)
            
            # Saturation (safe)
            if self.saturation_enabled.get():
                saturation = min(1.5, max(0.5, self.effect_params['saturation']))  # Limit range
                img = ImageEnhance.Color(img).enhance(saturation)
            
            # VERY gentle blur if enabled
            if self.blur_enabled.get() and self.effect_params['blur'] > 0:
                # Limit blur radius to prevent corruption
                blur_radius = min(0.5, self.effect_params['blur'])
                if blur_radius > 0.1:  # Only apply if significant enough
                    try:
                        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                    except Exception as blur_error:
                        print(f"Skipping blur effect: {blur_error}")
            
            # Apply hue shift if enabled and available
            if hasattr(self, 'hue_shift_enabled') and self.hue_shift_enabled.get() and 'hue_shift' in self.effect_params:
                if HAS_NUMPY and HAS_SKIMAGE:
                    try:
                        # Convert to HSV color space
                        hsv_img = img.convert('HSV')
                        h, s, v = hsv_img.split()
                        
                        # Apply slight hue rotation with proper bounds checking
                        shift_amount = self.effect_params['hue_shift']
                        h_data = np.array(h)
                    
                        # Calculate shift and ensure it stays within bounds
                        shift_pixels = int(shift_amount * 255/360)
                    
                        # Use numpy's clip and modulo to ensure values stay within 0-255 range
                        h_data = np.mod(h_data.astype(np.int16) + shift_pixels, 256).astype(np.uint8)
                    
                        h = Image.fromarray(h_data)
                    
                        # Recombine channels
                        hsv_img = Image.merge('HSV', (h, s, v))
                        img = hsv_img.convert('RGB')
                    except Exception as e:
                        print(f"Warning: Minor issue in hue processing (safe to ignore): {e}")
            
            # Apply noise if enabled
            if hasattr(self, 'noise_enabled') and self.noise_enabled.get() and 'noise' in self.effect_params and HAS_NUMPY:
                try:
                    noise_level = self.effect_params['noise']
                    img_array = np.array(img)
                    noise = np.random.normal(0, noise_level, img_array.shape)
                    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
                    img = Image.fromarray(noisy_img)
                except Exception as e:
                    print(f"Warning: Minor issue in noise application (safe to ignore): {e}")
            
            # Apply region effects if enabled
            if hasattr(self, 'region_enabled') and self.region_enabled.get() and hasattr(self, 'region_blend_var'):
                try:
                    # Only import ImageDraw if needed
                    from PIL import ImageDraw
                    
                    # Make a copy to avoid modifying the original
                    img_with_regions = img.copy()
                    width, height = img.size
                
                    # Get blend amount (0-100%)
                    blend_amount = self.region_blend_var.get() / 100.0
                
                    # If blend is significant, apply region effects
                    if blend_amount > 0.05:
                        # Create a mask for blending
                        mask = Image.new('L', (width, height), 128)  # Start with middle gray
                        draw = ImageDraw.Draw(mask)
                        
                        # Seed variant to ensure consistency
                        seed_variant = hash((self.effect_seed.get())) % 1000
                        random.seed(self.effect_seed.get() + seed_variant + frame_index)
                        
                        # Apply subtle random variations to different parts of the image
                        num_variations = 4  # Number of random regions
                        for i in range(num_variations):
                            # Random rectangle somewhere in the image
                            x1 = random.randint(0, width-10)
                            y1 = random.randint(0, height-10)
                            x2 = min(width, x1 + random.randint(width//4, width//2))
                            y2 = min(height, y1 + random.randint(height//4, height//2))
                            
                            # Random brightness variation - very subtle
                            variation = random.uniform(0.95, 1.05)
                            
                            # Mark this region on the mask - subtle
                            blend_val = int(128 + (variation - 1.0) * 50)  # 128 is neutral
                            draw.rectangle([x1, y1, x2, y2], fill=blend_val)
                        
                        # Apply the mask - blend between original and subtle variation
                        img = Image.blend(img, img_with_regions, blend_amount * 0.3)  # Reduce effect strength
                
                except Exception as e:
                    print(f"Warning: Error in region effects (safe to ignore): {e}")
            
            # SKIP rotation and resize - these cause artifacts
            
            # Ensure image stays in RGB mode
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # CRITICAL: Verify dimensions match the original frame
            if img.size != original_dimensions:
                img = img.resize(original_dimensions, Image.LANCZOS)
            
            # Apply invisible changes to make each frame unique
            img = self.apply_invisible_changes(img, frame_index)
                
            return img
            
        except Exception as e:
            print(f"Error applying effects: {e}")
            # On any error, return the original frame unchanged
            return frame.copy()
    
    def apply_invisible_changes(self, img, frame_index=0):
        """Apply invisible changes to make each frame unique without visible differences"""
        try:
            # Add a unique pixel modification that won't be visually detectable
            # This modifies a single pixel with a barely perceptible change based on frame index
            if img.mode == 'RGB':
                width, height = img.size
                
                # Choose positions that are unique to this GIF but consistent across frames
                x = (hash(self.unique_id) + frame_index) % max(1, width - 1)
                y = (hash(self.unique_id[::-1]) + frame_index) % max(1, height - 1)
                
                # Get current pixel
                r, g, b = img.getpixel((x, y))
                
                # Make a tiny change (±1) that's not visually perceptible
                r = max(0, min(255, r + ((frame_index + hash(self.unique_id)) % 3) - 1))
                g = max(0, min(255, g + ((frame_index + hash(self.unique_id[::-1])) % 3) - 1))
                b = max(0, min(255, b + ((frame_index + hash(self.unique_id + str(frame_index))) % 3) - 1))
                
                # Set the modified pixel
                img.putpixel((x, y), (r, g, b))
            
            # Add metadata that makes each GIF unique
            img.info.update({
                f"frame_id": f"{hash(self.unique_id) + frame_index}",
                f"timestamp": f"{time.time() + hash(self.unique_id) % 1000 + frame_index/1000}",
                f"unique": self.unique_id
            })
            
            return img
            
        except Exception as e:
            print(f"Error in invisible changes: {str(e)}")
            return img
    
    def verify_unique_changes(self, original_frame, processed_frame):
        """Verify that enough changes have been made to avoid detection"""
        try:
            if not HAS_NUMPY:
                # If numpy isn't available, we'll assume changes are sufficient
                return True
                
            # Convert images to numpy arrays for comparison
            import numpy as np
            from PIL import ImageChops
        
            # Calculate difference between frames
            diff = ImageChops.difference(original_frame, processed_frame)
        
            # Calculate percentage of changed pixels
            diff_array = np.array(diff)
            total_pixels = diff_array.size / 3  # RGB has 3 channels
            changed_pixels = np.count_nonzero(diff_array) / 3
        
            change_percentage = (changed_pixels / total_pixels) * 100
        
            # Log the amount of change
            print(f"Frame change percentage: {change_percentage:.2f}%")
        
            # Return True if changes exceed threshold (3%)
            return change_percentage >= 3.0
            
        except Exception as e:
            print(f"Error verifying changes: {str(e)}")
            return True  # Assume changes are sufficient if verification fails
    
    def on_scale_change(self, *args):
        """Handle scale change events"""
        # Ensure start time doesn't exceed end time
        start = self.start_time_var.get()
        end = self.end_time_var.get()

        if start > end:
            # Swap values if start > end
            self.start_time_var.set(end)
            self.end_time_var.set(start)
            start, end = end, start

        duration = end - start
        self.duration_var.set(duration)
        self.duration_label.config(text=f"{duration:.1f}")

        # Update video preview selection if available
        if self.video_preview and hasattr(self.video_preview, 'update_selection'):
            self.video_preview.update_selection(start, end)
    
    
    def _create_single_gif_direct(self, start_time, end_time, playback_speed, prefix=""):
        """Create a single GIF directly with FFmpeg for perfect timing"""
        try:
            duration = end_time - start_time
                
            # Create truly random filename with no recognizable pattern
            # Random elements for filename
            random_words = ["blue", "green", "sunny", "cloud", "happy", "smile", "jump", "dance", 
                            "wave", "star", "moon", "light", "dark", "cool", "warm", "swift", 
                            "calm", "soft", "bold", "play", "zoom", "flow", "spin", "flip"]
                                
            random_word1 = random.choice(random_words)
            random_word2 = random.choice(random_words)
            random_numbers = ''.join(random.choices('0123456789', k=4))
            random_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
              
            # Timestamp with some randomness
            timestamp = datetime.now().strftime("%H%M%S")
            timestamp = ''.join(random.sample(timestamp, len(timestamp)))  # Shuffle timestamp digits
                
            # Construct filename with no discernible pattern
            filename = f"{random_word1}_{random_numbers}_{random_word2}_{timestamp}_{random_id}.gif"
            output_path = os.path.join(self.output_folder, filename)
            
            # Create a temporary directory for frames
            temp_frames_dir = os.path.join(self.temp_dir, f"frames_{random_id}")
            os.makedirs(temp_frames_dir, exist_ok=True)
            
            # Step 1: Extract frames with high quality
            fps = 30  # Target frame rate
            max_width = self.max_width_var.get()
            
            # Simplify the scale filter to avoid parsing issues
            extract_cmd = [
                FFMPEG_PATH,
                '-ss', str(start_time),
                '-t', str(duration),
                '-i', self.input_path,
                '-vf', f'fps={fps},scale={max_width}:-1',  # Simple scaling
                '-q:v', '1',  # High quality
                os.path.join(temp_frames_dir, 'frame_%04d.jpg')
            ]
            
            # Run extraction
            print(f"Running FFmpeg command: {' '.join(extract_cmd)}")
            extract_process = subprocess.run(extract_cmd, capture_output=True)
            if extract_process.returncode != 0:
                error = extract_process.stderr.decode('utf-8', errors='ignore')
                raise Exception(f"Error extracting frames: {error}")
            
            # Step 2: Load frames and create GIF with PIL
            frames = []
            frame_files = sorted([f for f in os.listdir(temp_frames_dir) if f.startswith("frame_") and f.endswith(".jpg")])
            
            if not frame_files:
                raise Exception("No frames were extracted")
            
            # Calculate the EXACT frame duration needed to match the source duration
            frame_duration_ms = int((duration * 1000) / len(frame_files) / playback_speed)
            frame_duration_ms = max(20, min(frame_duration_ms, 200))  # Reasonable bounds
            
            # Load all frames with consistent dimensions
            reference_size = None
            for frame_file in frame_files:
                img = Image.open(os.path.join(temp_frames_dir, frame_file))
                
                # Record first frame size as reference
                if reference_size is None:
                    reference_size = img.size
                
                # Ensure RGB mode
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Ensure consistent size
                if img.size != reference_size:
                    img = img.resize(reference_size, Image.LANCZOS)
                
                frames.append(img)
            
            # Save GIF with exact frame durations
            if frames:
                frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=frames[1:],
                    optimize=False,
                    duration=frame_duration_ms,
                    loop=0 if self.loop_var.get() else 1
                )
                
                # Clean up
                shutil.rmtree(temp_frames_dir)
                
                print(f"Successfully created GIF: {output_path}")
                return output_path
            else:
                raise Exception("No frames were extracted")
            
        except Exception as e:
            print(f"Error in direct GIF creation: {e}")
            raise
    
    
    
    def _create_single_video_direct(self, start_time, end_time, playback_speed, prefix=""):
        """Create a single processed video directly with FFmpeg"""
        try:
            duration = end_time - start_time
                
            # Create truly random filename with no recognizable pattern
            # Random elements for filename
            random_words = ["blue", "green", "sunny", "cloud", "happy", "smile", "jump", "dance", 
                            "wave", "star", "moon", "light", "dark", "cool", "warm", "swift", 
                            "calm", "soft", "bold", "play", "zoom", "flow", "spin", "flip"]
                                
            random_word1 = random.choice(random_words)
            random_word2 = random.choice(random_words)
            random_numbers = ''.join(random.choices('0123456789', k=4))
            random_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
              
            # Timestamp with some randomness
            timestamp = datetime.now().strftime("%H%M%S")
            timestamp = ''.join(random.sample(timestamp, len(timestamp)))  # Shuffle timestamp digits
                
            # Construct filename with no discernible pattern
            video_format = self.video_format_var.get()
            filename = f"{random_word1}_{random_numbers}_{random_word2}_{timestamp}_{random_id}.{video_format}"
            output_path = os.path.join(self.output_folder, filename)
            
            # Create a temporary directory for processed frames
            temp_frames_dir = os.path.join(self.temp_dir, f"frames_{random_id}")
            os.makedirs(temp_frames_dir, exist_ok=True)
            
            # Step 1: Extract frames with high quality
            fps = 30  # Target frame rate
            max_width = self.max_width_var.get()
            
            # IMPORTANT: Make sure the height is even by using a specific scale with pad filter
            # This adds a vf filter to ensure height is even (required by h264)
            extract_cmd = [
                FFMPEG_PATH,
                '-ss', str(start_time),
                '-t', str(duration),
                '-i', self.input_path,
                '-vf', f'fps={fps},scale={max_width}:-2,pad=ceil(iw/2)*2:ceil(ih/2)*2',  # Scale with even height and pad if needed
                '-q:v', '1',  # High quality
                os.path.join(temp_frames_dir, 'frame_%04d.jpg')
            ]
            
            # Run extraction
            print(f"Running FFmpeg command: {' '.join(extract_cmd)}")
            extract_process = subprocess.run(extract_cmd, capture_output=True)
            if extract_process.returncode != 0:
                error = extract_process.stderr.decode('utf-8', errors='ignore')
                raise Exception(f"Error extracting frames: {error}")
            
            # Step 2: Load frames, apply effects, and save processed frames
            frames = []
            frame_files = sorted([f for f in os.listdir(temp_frames_dir) if f.startswith("frame_") and f.endswith(".jpg")])
            
            if not frame_files:
                raise Exception("No frames were extracted")
            
            # Process each frame with effects
            for i, frame_file in enumerate(frame_files):
                # Load frame
                img = Image.open(os.path.join(temp_frames_dir, frame_file))
                
                # Apply effects to frame
                if not hasattr(self, 'effect_params') or not self.consistent_effects.get():
                    self.apply_consistent_effects()
                
                processed_frame = self.apply_effects_to_frame(img, i)
                
                # Save processed frame
                processed_frame.save(os.path.join(temp_frames_dir, f"proc_{frame_file}"))
                
                # Update progress (50% of progress is frame processing)
                progress = (i / len(frame_files)) * 50
                self.progress_var.set(progress)
                
                if i % 10 == 0:  # Update status every 10 frames
                    self.status_var.set(f"Processing frame {i+1}/{len(frame_files)}")
                    self.root.update_idletasks()
            
            # Step 3: Create video from processed frames
            fps_adjusted = fps * playback_speed  # Adjust FPS based on playback speed
            codec = self.video_codec_var.get()
            bitrate = self.video_bitrate_var.get()
            
            # Build the FFmpeg command for video creation with simpler, more reliable settings
            video_cmd = [
                FFMPEG_PATH,
                '-framerate', str(fps_adjusted),
                '-i', os.path.join(temp_frames_dir, 'proc_frame_%04d.jpg'),
                '-c:v', 'libx264',  # More reliable than 'h264'
                '-preset', 'medium',
                '-pix_fmt', 'yuv420p',
                '-b:v', bitrate,
                '-y',
                output_path
            ]
            
            # Run video creation
            self.status_var.set(f"Creating final video...")
            self.root.update_idletasks()
            
            video_process = subprocess.run(video_cmd, capture_output=True)
            if video_process.returncode != 0:
                error = video_process.stderr.decode('utf-8', errors='ignore')
                raise Exception(f"Error creating video: {error}")
            
            # Clean up temp directory
            shutil.rmtree(temp_frames_dir)
            
            self.progress_var.set(100)
            self.status_var.set(f"Video created successfully: {filename}")
            
            return output_path
        
        except Exception as e:
            print(f"Error in direct video creation: {e}")
            self.status_var.set(f"Error creating video: {str(e)}")
            raise
    
    
    def start_processing(self):
        """Start processing frames into GIFs"""
        if self.is_processing:
            self.status_var.set("Processing already in progress")
            return
        
        # Check if in folder mode
        if hasattr(self, 'is_folder_mode') and self.is_folder_mode:
            self.start_bulk_processing()
            return
        
        # Individual file processing (original code)
        # Check how many copies to make
        num_copies = self.copies_var.get()
        copy_type = self.copy_type_var.get()
        
        if num_copies <= 0:
            self.status_var.set("Number of copies must be at least 1")
            return
        
        if copy_type == "exact":
            # Use the fixed direct method for exact copies
            threading.Thread(target=self.create_exact_copies, 
                           args=(num_copies,), daemon=True).start()
        else:  # random segments
            # Process random segments
            if not self.is_video or not self.input_path:
                self.status_var.set("Video required for random segments")
                return
                
            threading.Thread(target=self.process_random_segments, 
                           args=(num_copies,), daemon=True).start()
    
    def start_bulk_processing(self):
        """Start bulk processing for folder mode"""
        try:
            # Validate we have folder
            if not hasattr(self, 'input_folder') or not self.input_folder:
                self.status_var.set("No folder selected for bulk processing")
                return
            
            # Get parameters
            num_copies = self.copies_var.get()
            copy_type = self.copy_type_var.get()
            
            if num_copies <= 0:
                self.status_var.set("Number of copies must be at least 1")
                return
                
            # Start bulk processing in a thread
            threading.Thread(target=self.process_folder, 
                           args=(self.input_folder, num_copies, copy_type), 
                           daemon=True).start()
        
        except Exception as e:
            self.status_var.set(f"Error starting bulk processing: {str(e)}")
            print(f"Error details: {str(e)}")
    
    def process_folder(self, folder_path, num_copies, copy_type):
        """Process all video files in a folder"""
        try:
            self.is_processing = True
            
            # Get all video files in folder
            video_files = self.get_video_files(folder_path)
            if not video_files:
                self.status_var.set("No video files found in selected folder")
                self.is_processing = False
                return
            
            # Get current output mode
            output_mode = self.output_mode_var.get()
            
            # Update status
            self.status_var.set(f"Processing {len(video_files)} videos, {num_copies} copies each")
            self.progress_var.set(0)
            self.root.update_idletasks()
            
            # Process each video
            for video_index, video_path in enumerate(video_files):
                # Update status for current video
                filename = os.path.basename(video_path)
                self.status_var.set(f"Processing video {video_index+1}/{len(video_files)}: {filename}")
                self.progress_var.set((video_index / len(video_files)) * 100)
                self.root.update_idletasks()
                
                # Set the current video path
                self.input_path = video_path
                
                # Get video duration
                try:
                    duration_cmd = [
                        FFPROBE_PATH,
                        '-v', 'error', 
                        '-show_entries', 'format=duration', 
                        '-of', 'default=noprint_wrappers=1:nokey=1', 
                        video_path
                    ]
                    result = subprocess.run(duration_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if result.stdout.strip():
                        video_duration = float(result.stdout.strip())
                    else:
                        video_duration = 10.0  # Default
                except Exception:
                    video_duration = 10.0  # Default if error
                
                # Process current video based on copy type
                video_prefix = f"video{video_index+1}_"
                
                if copy_type == "exact":
                    # For exact copies, use the entire video duration
                    start_time = 0
                    end_time = video_duration
                    playback_speed = self.frame_duration_var.get()
                    
                    for copy_index in range(num_copies):
                        # Create a unique seed for this copy
                        seed = random.randint(1, 1000000)
                        self.effect_seed.set(seed)
                        
                        # Generate a unique prefix for this copy
                        copy_prefix = f"{video_prefix}copy{copy_index+1}_"
                        
                        try:
                            # Create output based on selected mode
                            if output_mode == "gif":
                                # Create GIF with direct method
                                self._create_single_gif_direct(
                                    start_time,
                                    end_time,
                                    playback_speed,
                                    copy_prefix
                                )
                            else:  # video mode
                                # Create processed video
                                self._create_single_video_direct(
                                    start_time,
                                    end_time,
                                    playback_speed,
                                    copy_prefix
                                )
                        except Exception as e:
                            print(f"Error creating copy {copy_index+1} of video {filename}: {e}")
                
                else:  # random segments
                    # Create multiple random segments from each video
                    for copy_index in range(num_copies):
                        # Get random segment parameters
                        min_length = self.random_min_length_var.get()
                        max_length = self.random_max_length_var.get()
                        
                        # Generate random segment length
                        segment_length = random.uniform(min_length, max_length)
                        if segment_length > video_duration:
                            segment_length = video_duration
                        
                        # Random start position
                        max_start = video_duration - segment_length
                        if max_start <= 0:
                            start_time = 0
                        else:
                            start_time = random.uniform(0, max_start)
                        
                        end_time = start_time + segment_length
                        playback_speed = self.frame_duration_var.get()
                        
                        # Create a unique seed for this segment
                        seed = random.randint(1, 1000000)
                        self.effect_seed.set(seed)
                        
                        # Generate a unique prefix for this segment
                        segment_prefix = f"{video_prefix}segment{copy_index+1}_"
                        
                        try:
                            # Create output based on selected mode
                            if output_mode == "gif":
                                # Create GIF with direct method
                                self._create_single_gif_direct(
                                    start_time,
                                    end_time,
                                    playback_speed,
                                    segment_prefix
                                )
                            else:  # video mode
                                # Create processed video
                                self._create_single_video_direct(
                                    start_time,
                                    end_time,
                                    playback_speed,
                                    segment_prefix
                                )
                        except Exception as e:
                            print(f"Error creating segment {copy_index+1} of video {filename}: {e}")
            
            # Update status when complete
            mode_text = "GIFs" if output_mode == "gif" else "Videos"
            self.status_var.set(f"Bulk processing complete! Processed {len(video_files)} videos into {mode_text}.")
            self.progress_var.set(100)
            
            # Show message
            self.root.after(0, lambda: messagebox.showinfo("Bulk Processing Complete", 
                          f"Successfully processed {len(video_files)} videos with {num_copies} copies each."))
            
        except Exception as e:
            self.status_var.set(f"Error in bulk processing: {str(e)}")
            print(f"Error in bulk processing: {str(e)}")
            
        finally:
            self.is_processing = False
    
    def get_video_files(self, folder_path):
        """Get list of video files in a folder"""
        video_extensions = ('.mp4', '.avi', '.mov', '.wmv', '.mkv', '.flv')
        video_files = []
        
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(file_path)
        
        return video_files
    
    def create_exact_copies(self, num_copies):
        """Create exact copies of the same segment with guaranteed timing preservation"""
        try:
            # Get timing information
            start_time = self.start_time_var.get()
            end_time = self.end_time_var.get()
            duration = end_time - start_time
            playback_speed = self.frame_duration_var.get()
            output_mode = self.output_mode_var.get()
            
            self.status_var.set(f"Creating {num_copies} identical copies...")
            self.progress_var.set(0)
            self.root.update_idletasks()
            
            # For each copy, do a completely fresh extraction and processing
            for i in range(num_copies):
                # Update progress
                progress = (i / num_copies) * 100
                self.progress_var.set(progress)
                self.status_var.set(f"Processing copy {i+1}/{num_copies} ({progress:.1f}%)...")
                self.root.update_idletasks()
                
                # Create unique seed for this copy
                unique_seed = random.randint(1, 1000000)
                random.seed(unique_seed)
                self.effect_seed.set(unique_seed)
                
                if output_mode == "gif":
                    # Create GIF
                    self._create_single_gif_direct(
                        start_time, 
                        end_time, 
                        playback_speed, 
                        f"copy{i+1}_"
                    )
                else:
                    # Create video
                    self._create_single_video_direct(
                        start_time,
                        end_time,
                        playback_speed,
                        f"copy{i+1}_"
                    )
            
            # Show appropriate completion message
            output_type = "GIFs" if output_mode == "gif" else "videos"
            self.status_var.set(f"Successfully created {num_copies} {output_type}")
            self.progress_var.set(100)
            messagebox.showinfo("Processing Complete", 
                             f"Successfully created {num_copies} exact copies of {duration:.2f} seconds!")
            
        except Exception as e:
            self.status_var.set(f"Error creating copies: {str(e)}")
            print(f"Error details: {str(e)}")
    
    def process_random_segments(self, num_copies):
        """Process random segments from video"""
        try:
            self.is_processing = True
            output_mode = self.output_mode_var.get()
            self.status_var.set(f"Creating {num_copies} copies from random segments...")
            self.progress_var.set(0)
            self.root.update_idletasks()
            
            # Get min and max segment lengths
            min_length = self.random_min_length_var.get()
            max_length = self.random_max_length_var.get()
            playback_speed = self.frame_duration_var.get()
            
            # Ensure valid lengths
            if min_length > max_length:
                min_length, max_length = max_length, min_length
            
            # Ensure minimum length is at least 1 second
            min_length = max(0.5, min_length)
            
            # Generate random segments
            segments = []
            for i in range(num_copies):
                # Random length between min and max
                seg_length = random.uniform(min_length, max_length)
                
                # Ensure it fits within video
                video_duration = self.video_duration
                if seg_length > video_duration:
                    seg_length = video_duration
                
                # Random start time
                max_start = video_duration - seg_length
                start_time = random.uniform(0, max_start if max_start > 0 else 0)
                end_time = start_time + seg_length
                
                segments.append((start_time, end_time))
            
            # Process each segment
            for i, (start_time, end_time) in enumerate(segments):
                # Update progress
                progress = ((i) / num_copies) * 100
                self.progress_var.set(progress)
                self.status_var.set(f"Processing segment {i+1}/{num_copies} ({progress:.1f}%)")
                self.root.update_idletasks()
                
                # Generate a new seed for each segment
                self.effect_seed.set(random.randint(1, 1000000))
                
                try:
                    if output_mode == "gif":
                        # Create GIF
                        self._create_single_gif_direct(
                            start_time,
                            end_time,
                            playback_speed,
                            f"random{i+1}_"
                        )
                    else:
                        # Create video
                        self._create_single_video_direct(
                            start_time,
                            end_time,
                            playback_speed,
                            f"random{i+1}_"
                        )
                except Exception as segment_error:
                    print(f"Error processing segment {i+1}: {segment_error}")
                    # Continue with next segment
            
            # Show appropriate completion message
            output_type = "GIFs" if output_mode == "gif" else "videos"
            self.status_var.set(f"Completed processing {num_copies} random segments")
            self.progress_var.set(100)
            messagebox.showinfo("Processing Complete", 
                              f"Successfully created {num_copies} {output_type} from random segments!")
            
        except Exception as e:
            self.status_var.set(f"Error processing random segments: {str(e)}")
            print(f"Error details: {str(e)}")
            
        finally:
            self.is_processing = False
    
    def process_frames(self, prefix=""):
        """Process all frames and create the GIF with high quality consistency"""
        try:
            # Save original frame count for verification
            original_frame_count = len(self.frames)
            if original_frame_count == 0:
                raise ValueError("No frames to process")
                
            self.is_processing = True
            self.status_var.set("Processing frames...")
            self.progress_var.set(0)
            self.root.update_idletasks()
            
            # Reset processed frames
            self.processed_frames = []
            
            # Generate the consistent effect parameters once for all frames
            if self.consistent_effects.get():
                self.apply_consistent_effects()
            
            # Get original frame dimensions to ensure consistency
            first_frame_dimensions = self.frames[0].size
            print(f"Original frame dimensions: {first_frame_dimensions}")
            
            # Process each frame
            for i, frame in enumerate(self.frames):
                try:
                    # Create a new copy of the frame to avoid reference issues
                    frame_copy = frame.copy()
                    
                    # Apply effects
                    processed_frame = self.apply_effects_to_frame(frame_copy, i)
                    
                    # CRITICAL: Ensure all frames have EXACTLY the same dimensions
                    if processed_frame.size != first_frame_dimensions:
                        print(f"Resizing frame {i} from {processed_frame.size} to {first_frame_dimensions}")
                        processed_frame = processed_frame.resize(first_frame_dimensions, Image.LANCZOS)
                    
                    # Add to processed frames list
                    self.processed_frames.append(processed_frame)
                    
                    # Update progress
                    progress = ((i + 1) / len(self.frames)) * 100
                    self.progress_var.set(progress)
                    
                    if i % 5 == 0:  # Update status every 5 frames
                        self.status_var.set(f"Processing frame {i+1}/{len(self.frames)} ({progress:.1f}%)")
                        self.root.update_idletasks()
                except Exception as frame_error:
                    print(f"Error processing frame {i}: {str(frame_error)}")
                    # Use original frame as fallback
                    self.processed_frames.append(frame.copy())
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            random_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=12))
            filename = f"{prefix}GIF_{timestamp}_{random_id}.gif"
            output_path = os.path.join(self.output_folder, filename)
            
            self.status_var.set("Creating GIF...")
            self.root.update_idletasks()
            
            # Apply frame transformations based on options
            frames_to_save = self.processed_frames.copy()
            
            # Reverse if selected
            if self.reverse_var.get():
                frames_to_save = frames_to_save[::-1]
            
            # Ping-pong if selected
            if self.ping_pong_var.get():
                # Add frames in reverse order, but skip duplicating first and last frames
                frames_to_save = frames_to_save + frames_to_save[-2:0:-1]
            
            try:
                # Calculate frame duration for consistency
                if hasattr(self, '_exact_frame_duration_ms'):
                    # Use stored exact frame duration
                    frame_duration_ms = self._exact_frame_duration_ms
                    
                    # Apply playback speed
                    speed_factor = self.frame_duration_var.get()
                    frame_duration_ms = int(frame_duration_ms / speed_factor)
                else:
                    # Default calculation based on frames
                    duration = end_time - start_time if hasattr(self, 'end_time') and hasattr(self, 'start_time') else 5.0
                    frame_duration_ms = int((duration * 1000) / len(frames_to_save))
                    
                    # Apply playback speed
                    speed_factor = self.frame_duration_var.get()
                    frame_duration_ms = int(frame_duration_ms / speed_factor)
                
                # Ensure reasonable bounds (10-200ms)
                frame_duration_ms = max(10, min(frame_duration_ms, 200))
                print(f"Using frame duration: {frame_duration_ms}ms per frame")
                
                # FINAL VALIDATION: Ensure all frames have identical dimensions
                first_dims = frames_to_save[0].size
                for i, frame in enumerate(frames_to_save):
                    if frame.size != first_dims:
                        print(f"Fixing dimensions for frame {i}: {frame.size} -> {first_dims}")
                        frames_to_save[i] = frame.resize(first_dims, Image.LANCZOS)
                
                # Save the GIF
                frames_to_save[0].save(
                    output_path,
                    save_all=True,
                    append_images=frames_to_save[1:],
                    optimize=False,  # Don't optimize to preserve quality
                    duration=frame_duration_ms,
                    loop=0 if self.loop_var.get() else 1,  # 0 = infinite loop
                    quality=self.quality_var.get()
                )
                print(f"Successfully saved GIF to: {output_path}")
                
                # Add metadata to make the GIF unique
                self.add_gif_metadata(output_path)
                
                self.status_var.set(f"GIF saved: {filename}")
            except Exception as gif_error:
                self.status_var.set(f"Error saving GIF: {str(gif_error)}")
                print(f"GIF save error: {str(gif_error)}")
            
            self.progress_var.set(100)
            
            # Show message if not in auto mode
            if not prefix:
                messagebox.showinfo("Success", f"GIF created successfully!\nSaved to: {output_path}")
                
        except Exception as e:
            self.status_var.set(f"Error creating GIF: {str(e)}")
            print(f"Error details: {str(e)}")
        
        finally:
            self.is_processing = False
    
    def add_gif_metadata(self, gif_path):
        """Add detailed metadata to the GIF file to make it unique"""
        try:
            # Original metadata
            metadata = {
                "creator": f"GifWasher_{random.getrandbits(32)}",
                "timestamp": str(time.time() + random.random()),
                "device": random.choice(["iPhone", "Samsung", "Google", "Huawei", "OnePlus"]),
                "processing_id": str(random.getrandbits(64)),
                "unique_hash": ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=16)),
                "camera": random.choice(["front", "rear", "wide", "ultra"]),
                "location": f"{random.uniform(-90, 90):.6f},{random.uniform(-180, 180):.6f}",
                "software": f"GifWasher{random.randint(1, 9)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
                "model": f"X{random.randint(1000, 9999)}",
                "processing": {
                    "brightness": random.uniform(0.8, 1.2),
                    "contrast": random.uniform(0.8, 1.2),
                    "quality": random.randint(85, 95),
                }
            }
            
            # Add a complex structure with nested data
            with open(gif_path, 'ab') as f:
                f.write(b'<!--')
                f.write(json.dumps(metadata, indent=2).encode('utf-8'))
                f.write(b'-->')
                
        except Exception as e:
            print(f"Warning: Minor issue with metadata (safe to ignore): {e}")
    
    def auto_generate_gifs(self):
        """Auto-generate multiple GIFs from the video"""
        try:
            self.is_processing = True
            self.status_var.set("Starting auto GIF generation...")
            self.progress_var.set(0)
            self.root.update_idletasks()
            
            # Get parameters
            num_gifs = self.num_gifs_var.get()
            min_length = self.min_gif_length_var.get()
            max_length = self.max_gif_length_var.get()
            playback_speed = self.frame_duration_var.get()
            
            # Calculate number of segments
            video_duration = self.video_duration
            
            # Generate random segments
            segments = []
            for i in range(num_gifs):
                # Random length between min and max
                seg_length = random.uniform(min_length, max_length)
                
                # Don't exceed video duration
                if seg_length > video_duration:
                    seg_length = video_duration
                
                # Random start time (ensure it fits within video)
                max_start = video_duration - seg_length
                start_time = random.uniform(0, max_start if max_start > 0 else 0)
                end_time = start_time + seg_length
                
                segments.append((start_time, end_time))
            
            # Process each segment
            for i, (start_time, end_time) in enumerate(segments):
                self.status_var.set(f"Processing segment {i+1}/{num_gifs} ({start_time:.1f}s - {end_time:.1f}s)")
                self.progress_var.set((i / num_gifs) * 100)
                self.root.update_idletasks()
                
                # Generate a new seed for each segment
                self.effect_seed.set(random.randint(1, 1000000))
                
                # Create GIF directly with FFmpeg
                try:
                    self._create_single_gif_direct(
                        start_time,
                        end_time,
                        playback_speed,
                        f"auto_{i+1}_"
                    )
                except Exception as segment_error:
                    print(f"Error processing segment {i+1}: {segment_error}")
                    # Continue with next segment
            
            self.status_var.set(f"Completed auto-generation of {num_gifs} GIFs")
            self.progress_var.set(100)
            messagebox.showinfo("Auto Generation Complete", 
                              f"Successfully created {num_gifs} GIFs from the video.")
            
        except Exception as e:
            self.status_var.set(f"Error in auto generation: {str(e)}")
            print(f"Error details: {str(e)}")
            
        finally:
            self.is_processing = False
    
    def extract_random_segment(self):
        """Extract a random segment from a longer video"""
        try:
                if not self.is_video or not self.input_path:
                        self.status_var.set("No video loaded")
                        return
                        
                self.status_var.set("Extracting random segment...")
                
                # Get video duration
                video_duration = self.video_duration
                
                # Determine segment length (between 3-7 seconds)
                segment_length = random.uniform(3.0, 7.0)
                
                # Ensure it fits within video
                if segment_length > video_duration:
                        segment_length = video_duration
                
                # Random start time
                max_start = video_duration - segment_length
                start_time = random.uniform(0, max_start)
                end_time = start_time + segment_length
                
                # Update UI
                self.start_time_var.set(start_time)
                self.end_time_var.set(end_time)
                self.update_duration()
                
                # Extract frames for this segment
                self.extract_frames()
                
                self.status_var.set(f"Extracted random segment ({start_time:.1f}s - {end_time:.1f}s)")
                
        except Exception as e:
                self.status_var.set(f"Error extracting random segment: {str(e)}")
                print(f"Error details: {str(e)}")
  
    def extract_frames_for_segment(self, start_time, end_time):
        """Extract frames for a specific video segment with high quality"""
        try:
            # Reset frames
            self.frames = []
            
            # Calculate duration
            duration = end_time - start_time
            print(f"Extracting segment: {start_time:.3f}s to {end_time:.3f}s (duration: {duration:.3f}s)")
            
            # Clear temp directory of old frames
            for file in os.listdir(self.temp_dir):
                if file.startswith("frame_"):
                    os.remove(os.path.join(self.temp_dir, file))
            
            # Use consistent 30fps for high quality
            target_fps = 30
            # Calculate expected number of frames for this duration
            total_frames = max(1, int(duration * target_fps))
            print(f"Targeting exactly {total_frames} frames at {target_fps}fps")
            
            # Fix the scale filter syntax to avoid the error - use simple scaling
            max_width = self.max_width_var.get()
            cmd = [
                FFMPEG_PATH,
                '-i', self.input_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-qscale:v', '1',      # Maximum quality
                '-vf', f'fps={target_fps},scale={max_width}:-1',  # Simple scale syntax
                os.path.join(self.temp_dir, 'frame_%04d.jpg')  # Using jpg for better compatibility
            ]
            
            # Run FFmpeg with proper error handling
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                error = stderr.decode('utf-8', errors='ignore')
                print(f"FFmpeg error: {error}")
                raise Exception(f"Failed to extract frames: {error}")
            
            # Load frames with extreme care
            frame_files = sorted([f for f in os.listdir(self.temp_dir) if f.startswith("frame_")])
            
            if not frame_files:
                raise Exception("No frames were extracted")
            
            # Calculate exact frame duration needed for GIF
            frame_duration_ms = int((duration * 1000) / len(frame_files))
            
            # Store for later use in all GIFs
            self._exact_frame_count = len(frame_files)
            self._exact_frame_duration_ms = frame_duration_ms
            
            print(f"Extracted exactly {len(frame_files)} frames")
            print(f"Each frame should display for {frame_duration_ms}ms to match {duration}s duration")
            
            # Store the reference dimensions of the first frame
            first_frame = None
            
            for frame_file in frame_files:
                try:
                    img_path = os.path.join(self.temp_dir, frame_file)
                    img = Image.open(img_path)
                    
                    # Immediately force load and convert to RGB to prevent any later issues
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    img.load()  # Force load image data now
                    
                    # Store first frame dimensions
                    if first_frame is None:
                        first_frame = img
                        self._frame_reference_dimensions = img.size
                    else:
                        # Ensure all frames have the exact same dimensions
                        if img.size != self._frame_reference_dimensions:
                            img = img.resize(self._frame_reference_dimensions, Image.LANCZOS)
                    
                    self.frames.append(img)
                except Exception as e:
                    print(f"Error loading frame {frame_file}: {e}")
            
            return len(self.frames)
            
        except Exception as e:
            print(f"ERROR extracting frames: {e}")
            # Ensure temp dir exists for future operations
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir)
            return 0
    
    def export_frames_as_images(self):
        """Export frames as individual image files"""
        try:
                if not self.frames:
                        self.status_var.set("No frames to export")
                        return
                        
                # Create a subfolder for the images
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                frames_folder = os.path.join(self.output_folder, f"frames_{timestamp}")
                if not os.path.exists(frames_folder):
                        os.makedirs(frames_folder)
                        
                self.status_var.set("Exporting frames as images...")
                self.progress_var.set(0)
                self.root.update_idletasks()
                
                # Export each frame
                for i, frame in enumerate(self.processed_frames if self.processed_frames else self.frames):
                        # Generate random filename similar to camera photos
                        camera_models = ["iPhone14Pro", "GalaxyS22", "Pixel7", "Canon_EOS_R5", "SonyA7IV", 
                                       "iPhone13", "GalaxyNote20", "Pixel6a", "NikonZ6", "GoPro11"]
                        location_tags = ["beach", "home", "park", "vacation", "party", "trip", "event", "family", "friends", "sunset"]
                        
                        random_device = random.choice(camera_models)
                        random_location = random.choice(location_tags)
                        random_date = datetime.now() - timedelta(
                                seconds=random.randint(1, 300)  # Small time difference between frames
                        )
                        date_string = random_date.strftime("%Y%m%d_%H%M%S")
                        random_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))
                        
                        filename = f"IMG_{date_string}_{random_device}_{random_location}_{random_id}.jpg"
                        output_path = os.path.join(frames_folder, filename)
                        
                        # Save the frame
                        frame.save(output_path, quality=95)
                        
                        # Update progress
                        progress = ((i + 1) / len(self.frames)) * 100
                        self.progress_var.set(progress)
                        
                        if i % 5 == 0:  # Update status every 5 frames
                                self.status_var.set(f"Exporting frame {i+1}/{len(self.frames)} ({progress:.1f}%)")
                                self.root.update_idletasks()
                                
                self.status_var.set(f"Exported {len(self.frames)} frames to {frames_folder}")
                messagebox.showinfo("Export Complete", f"Frames exported successfully to:\n{frames_folder}")
                
        except Exception as e:
                self.status_var.set(f"Error exporting frames: {str(e)}")
                print(f"Error details: {str(e)}")
    
    def bulk_export_frames_as_images(self):
        """Export frames from all videos in a folder as individual image files"""
        try:
            # Check if in folder mode
            if not hasattr(self, 'is_folder_mode') or not self.is_folder_mode or not self.input_folder:
                self.status_var.set("Please select a folder of videos first")
                return
                
            # Confirm with user
            response = messagebox.askquestion("Bulk Export Frames", 
                                             f"This will extract frames from all videos in the folder. Continue?",
                                             icon='question')
            if response != 'yes':
                return
                
            # Start processing in a thread
            threading.Thread(target=self._bulk_export_frames_thread, 
                           args=(self.input_folder,), 
                           daemon=True).start()
                
        except Exception as e:
            self.status_var.set(f"Error starting bulk export: {str(e)}")
            print(f"Error details: {str(e)}")
    
    
    def open_output_folder(self):
        """Open the output folder in file explorer"""
        output_path = self.output_folder
        if os.path.exists(output_path):
            # Open folder based on platform
            if sys.platform == 'win32':
                os.startfile(output_path)
            elif sys.platform == 'darwin':  # macOS
                os.system(f'open "{output_path}"')
            else:  # Linux
                os.system(f'xdg-open "{output_path}"')
        else:
            self.status_var.set("Output folder does not exist")
    
    def on_closing(self):
        """Handle window closing event"""
        try:
            # Clean up video preview if any
            if hasattr(self, 'video_preview') and self.video_preview:
                self.video_preview.cleanup()
                
            # Clean up temp directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Error cleaning up: {str(e)}")
        
        # Close application
        self.root.destroy()
    
    def run(self):
        """Run the application"""
        self.root.mainloop()


def check_ffmpeg():
    """Check if FFmpeg binaries are available in the application directory"""
    if not os.path.exists(FFMPEG_PATH) or not os.path.exists(FFPROBE_PATH):
        return False
    return True


if __name__ == "__main__":
    try:
        app = AdvancedGifWasher()
        app.run()
    except Exception as e:
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")  # This prevents immediate window closing
        sys.exit(1)
        
