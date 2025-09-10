#!/usr/bin/env python3
"""
Example usage of timing debug utilities.

This file shows how to use the timing debug tools in different scenarios.
"""

from verl.utils.timing_debug import (
    print_timing, 
    start_timer, 
    end_timer, 
    context_timer, 
    timing_decorator,
    set_timing_debug,
    TimingDebugger
)
import time
import torch.distributed


def example_basic_usage():
    """Example of basic timing functions"""
    
    # Enable timing debug (default is enabled)
    set_timing_debug(enable=True, rank_filter=0)
    
    # Simple step timing
    print_timing("Starting data loading")
    time.sleep(0.1)  # Simulate work
    print_timing("Finished data loading")
    
    # Named timer
    start_timer("forward_pass")
    time.sleep(0.2)  # Simulate work
    duration = end_timer("forward_pass")
    print(f"Forward pass took {duration:.3f}s")


def example_context_manager():
    """Example of context manager usage"""
    
    with context_timer("model_forward"):
        time.sleep(0.15)  # Simulate work
        
        with context_timer("attention_computation"):
            time.sleep(0.05)  # Simulate work
    
    # Nested context managers work well for hierarchical timing


@timing_decorator("my_training_step")
def example_function():
    """Example of decorator usage"""
    time.sleep(0.1)  # Simulate work
    return "success"


def example_custom_debugger():
    """Example of using custom debugger instance"""
    
    # Create custom debugger for specific component
    router_debugger = TimingDebugger(rank_filter=0, enable=True)
    
    router_debugger.print_timing("Starting router logits processing")
    
    with router_debugger.context_timer("router_aggregation"):
        time.sleep(0.08)  # Simulate work
    
    # Use decorator from custom debugger
    @router_debugger.timing_decorator("router_broadcast")
    def broadcast_router_logits():
        time.sleep(0.05)  # Simulate work
    
    broadcast_router_logits()


def example_conditional_timing():
    """Example of conditional timing based on environment"""
    
    # Only enable for debugging
    debug_mode = True  # Could be from env var or config
    set_timing_debug(enable=debug_mode)
    
    if debug_mode:
        print_timing("Debug mode enabled - detailed timing")
    
    start_timer("conditional_work")
    time.sleep(0.1)
    end_timer("conditional_work")


if __name__ == "__main__":
    print("=== Timing Debug Examples ===")
    
    print("\n1. Basic usage:")
    example_basic_usage()
    
    print("\n2. Context manager:")
    example_context_manager()
    
    print("\n3. Function decorator:")
    result = example_function()
    print(f"Function returned: {result}")
    
    print("\n4. Custom debugger:")
    example_custom_debugger()
    
    print("\n5. Conditional timing:")
    example_conditional_timing()
    
    print("\n=== Examples completed ===")
