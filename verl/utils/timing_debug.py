# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Timing debug utilities for performance analysis and bottleneck identification.
"""

import time
from datetime import datetime
from functools import wraps
from typing import Optional, Dict, Any
import torch.distributed


class TimingDebugger:
    """A class to handle timing debug functionality"""
    
    def __init__(self, rank_filter: int = 0, enable: bool = True):
        """
        Initialize timing debugger
        
        Args:
            rank_filter: Only print from this rank (default: rank 0)
            enable: Whether timing debug is enabled
        """
        self.rank_filter = rank_filter
        self.enable = enable
        self.timers = {}
    
    def print_timing(self, message: str, rank_filter: Optional[int] = None):
        """Print timing information with current timestamp
        
        Args:
            message: Description of the current step
            rank_filter: Override default rank filter for this print
        """
        if not self.enable:
            return
            
        target_rank = rank_filter if rank_filter is not None else self.rank_filter
        
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != target_rank:
            return
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[TIMING {current_time}] {message}", flush=True)
    
    def start_timer(self, name: str):
        """Start a named timer"""
        if not self.enable:
            return
        self.timers[name] = time.time()
        self.print_timing(f"Timer '{name}' started")
    
    def end_timer(self, name: str) -> float:
        """End a named timer and return duration"""
        if not self.enable:
            return 0.0
            
        if name not in self.timers:
            self.print_timing(f"Timer '{name}' not found!")
            return 0.0
        
        duration = time.time() - self.timers[name]
        self.print_timing(f"Timer '{name}' finished in {duration:.3f}s")
        del self.timers[name]
        return duration
    
    def timing_decorator(self, func_name: Optional[str] = None):
        """Decorator to automatically time function execution
        
        Args:
            func_name: Optional custom name for the function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enable:
                    return func(*args, **kwargs)
                
                name = func_name or f"{func.__module__}.{func.__name__}"
                start_time = time.time()
                self.print_timing(f"Starting {name}")
                
                try:
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    self.print_timing(f"Finished {name} in {end_time - start_time:.3f}s")
                    return result
                except Exception as e:
                    end_time = time.time()
                    self.print_timing(f"Failed {name} after {end_time - start_time:.3f}s: {e}")
                    raise
            return wrapper
        return decorator
    
    def context_timer(self, name: str):
        """Context manager for timing code blocks"""
        return TimingContext(self, name)


class TimingContext:
    """Context manager for timing code blocks"""
    
    def __init__(self, debugger: TimingDebugger, name: str):
        self.debugger = debugger
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        if self.debugger.enable:
            self.start_time = time.time()
            self.debugger.print_timing(f"Starting '{self.name}'")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.debugger.enable and self.start_time is not None:
            duration = time.time() - self.start_time
            if exc_type is None:
                self.debugger.print_timing(f"Finished '{self.name}' in {duration:.3f}s")
            else:
                self.debugger.print_timing(f"Failed '{self.name}' after {duration:.3f}s: {exc_val}")


# Global default instance
_default_debugger = TimingDebugger()


def print_timing(message: str, rank_filter: Optional[int] = None):
    """Global function to print timing information
    
    Args:
        message: Description of the current step
        rank_filter: Only print from this rank (default: rank 0)
    """
    _default_debugger.print_timing(message, rank_filter)


def start_timer(name: str):
    """Start a named timer using global debugger"""
    _default_debugger.start_timer(name)


def end_timer(name: str) -> float:
    """End a named timer using global debugger"""
    return _default_debugger.end_timer(name)


def timing_decorator(func_name: Optional[str] = None):
    """Decorator to automatically time function execution using global debugger"""
    return _default_debugger.timing_decorator(func_name)


def context_timer(name: str):
    """Context manager for timing code blocks using global debugger"""
    return _default_debugger.context_timer(name)


def set_timing_debug(enable: bool, rank_filter: int = 0):
    """Enable/disable timing debug globally
    
    Args:
        enable: Whether to enable timing debug
        rank_filter: Only print from this rank
    """
    global _default_debugger
    _default_debugger.enable = enable
    _default_debugger.rank_filter = rank_filter


# Convenience functions for common use cases
def time_step(step_name: str):
    """Print timing for a step"""
    print_timing(f"Step: {step_name}")


def time_section_start(section_name: str):
    """Start timing a section"""
    start_timer(section_name)


def time_section_end(section_name: str):
    """End timing a section"""
    return end_timer(section_name)
