import h5py
import numpy as np
from pathlib import Path
import time

def add_samples_to_hdf5():
    current_dir = Path(__file__).parent
    hdf5_path = current_dir / 'robot_data.hdf5'
    
    # Open the existing file in read/write mode
    with h5py.File(hdf5_path, 'r+') as f:
        # Get the current number of samples
        data_group = f['data']
        current_samples = f.attrs['total_samples']
        new_samples = 10
        total_samples = current_samples + new_samples
        
        # Generate new data
        for i in range(current_samples, total_samples):
            # Create new sample group
            sample_group = data_group.create_group(f'sample_{i:05d}')
            
            # Generate sample data
            task_desc = f"Pick up the cup from the kitchen counter and place it on table {i}"
            task_embedding = np.random.rand(512)  # 512-dim embedding
            rgb_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            depth_img = np.random.rand(480, 640).astype(np.float32)
            object_mask = np.random.randint(0, 10, (480, 640), dtype=np.uint8)
            robot_joints = np.random.rand(7)  # 7 joint angles
            gripper_pose = np.random.rand(7)  # [x, y, z, qx, qy, qz, qw]
            
            # Store data
            sample_group.create_dataset('task_description', 
                                      data=task_desc.encode('utf-8'))
            
            sample_group.create_dataset('task_embedding', 
                                      data=task_embedding)
            
            sample_group.create_dataset('rgb', 
                                      data=rgb_img,
                                      compression='gzip',
                                      compression_opts=9)
            
            sample_group.create_dataset('depth', 
                                      data=depth_img,
                                      compression='gzip',
                                      compression_opts=9)
            
            sample_group.create_dataset('object_mask', 
                                      data=object_mask,
                                      compression='gzip',
                                      compression_opts=9)
            
            sample_group.create_dataset('robot_joints', 
                                      data=robot_joints)
            
            sample_group.create_dataset('gripper_pose', 
                                      data=gripper_pose)
            
            # Add metadata
            sample_group.attrs['timestamp'] = str(np.datetime64('now')).encode('utf-8')
            
            print(f"Added sample {i:05d}")
        
        # Update total samples count
        f.attrs['total_samples'] = total_samples
        print(f"\nSuccessfully added {new_samples} new samples.")
        print(f"Total samples in dataset: {total_samples}")

if __name__ == '__main__':
    try:
        add_samples_to_hdf5()
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure the HDF5 file exists and is not being used by another program.") 