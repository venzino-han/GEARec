#!/usr/bin/env python3
"""
Device Tracker - Utility for tracking GPU usage status
Tracks and manages which device each run is executing on.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


class DeviceTracker:
    def __init__(self, tracking_file="device_status.json"):
        """
        Args:
            tracking_file: file path to save device status
        """
        # Find rl4rec project root directory
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        self.tracking_file = project_root / tracking_file
        
    def _load_status(self):
        """Load current device status"""
        if not self.tracking_file.exists():
            return {}
        
        try:
            with open(self.tracking_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    
    def _save_status(self, status):
        """Save device status"""
        # Create directory if it does not exist
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.tracking_file, 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=2, ensure_ascii=False)
    
    def allocate(self, device_id, run_name):
        """
        Allocate a device to a specific run

        Args:
            device_id: GPU device ID (0, 1, 2, ...)
            run_name: name of the run to execute
        """
        status = self._load_status()
        device_key = f"cuda:{device_id}"
        
        status[device_key] = {
            "run_name": run_name,
            "status": "running",
            "start_time": datetime.now().isoformat(),
            "device_id": device_id
        }
        
        self._save_status(status)
        print(f"✅ Device {device_id} allocated to: {run_name}")
    
    def free(self, device_id):
        """
        Set a device to free status

        Args:
            device_id: GPU device ID (0, 1, 2, ...)
        """
        status = self._load_status()
        device_key = f"cuda:{device_id}"
        
        if device_key in status:
            old_run = status[device_key].get("run_name", "unknown")
            start_time = status[device_key].get("start_time", "unknown")
            
            status[device_key] = {
                "run_name": "free",
                "status": "free",
                "last_run": old_run,
                "last_freed": datetime.now().isoformat(),
                "last_start_time": start_time,
                "device_id": device_id
            }
        else:
            status[device_key] = {
                "run_name": "free",
                "status": "free",
                "last_freed": datetime.now().isoformat(),
                "device_id": device_id
            }
        
        self._save_status(status)
        print(f"✅ Device {device_id} is now free (previous run: {old_run if device_key in status else 'none'})")
    
    def show(self):
        """Display current status of all devices"""
        status = self._load_status()
        
        if not status:
            print("No devices tracked yet.")
            return
        
        print("\n" + "="*80)
        print("GPU Device Status")
        print("="*80)
        
        for device_key in sorted(status.keys()):
            info = status[device_key]
            device_id = info.get("device_id", "?")
            run_name = info.get("run_name", "unknown")
            device_status = info.get("status", "unknown")
            
            if device_status == "running":
                start_time = info.get("start_time", "unknown")
                print(f"\n🔴 Device {device_id} (cuda:{device_id}): RUNNING")
                print(f"   Run: {run_name}")
                print(f"   Started: {start_time}")
            else:
                last_freed = info.get("last_freed", "unknown")
                last_run = info.get("last_run", "none")
                print(f"\n🟢 Device {device_id} (cuda:{device_id}): FREE")
                print(f"   Last run: {last_run}")
                print(f"   Freed at: {last_freed}")
        
        print("\n" + "="*80 + "\n")
    
    def show_simple(self):
        """Display device status in simple format (use before/after script execution)"""
        status = self._load_status()
        
        if not status:
            print("📊 No devices tracked yet.")
            return
        
        print("\n📊 Current GPU Status:")
        for device_key in sorted(status.keys()):
            info = status[device_key]
            device_id = info.get("device_id", "?")
            run_name = info.get("run_name", "unknown")
            device_status = info.get("status", "unknown")
            
            if device_status == "running":
                print(f"   🔴 GPU {device_id}: {run_name}")
            else:
                print(f"   🟢 GPU {device_id}: free")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="GPU Device Tracker - track which runs are using which devices"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # allocate command
    allocate_parser = subparsers.add_parser('allocate', help='Allocate device to a run')
    allocate_parser.add_argument('device_id', type=int, help='GPU device ID (e.g., 0, 1, 2)')
    allocate_parser.add_argument('run_name', type=str, help='Name of the run')
    
    # free command
    free_parser = subparsers.add_parser('free', help='Free a device')
    free_parser.add_argument('device_id', type=int, help='GPU device ID to free')
    
    # show command
    subparsers.add_parser('show', help='Show all device statuses')
    
    # show-simple command
    subparsers.add_parser('show-simple', help='Show device statuses in simple format')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    tracker = DeviceTracker()
    
    if args.command == 'allocate':
        tracker.allocate(args.device_id, args.run_name)
    elif args.command == 'free':
        tracker.free(args.device_id)
    elif args.command == 'show':
        tracker.show()
    elif args.command == 'show-simple':
        tracker.show_simple()


if __name__ == '__main__':
    main()

