import h5py
import numpy as np
from pathlib import Path
current_dir = Path(__file__).parent

def create_and_write_hdf5():
    # Create sample data
    num_samples = 100
    
    # Generate data
    task_descriptions = [f"Pick up the cup and place it on the table {i}" for i in range(num_samples)]
    task_embeddings = np.random.rand(num_samples, 512)  # 512-dim embedding
    rgb_images = np.random.randint(0, 255, (num_samples, 480, 640, 3), dtype=np.uint8)  # 480x640 RGB
    depth_images = np.random.rand(num_samples, 480, 640).astype(np.float32)  # 480x640 depth
    object_masks = np.random.randint(0, 10, (num_samples, 480, 640), dtype=np.uint8)  # 480x640 mask
    robot_joints = np.random.rand(num_samples, 7)  # 7 joint angles
    gripper_poses = np.random.rand(num_samples, 7)  # [x, y, z, qx, qy, qz, qw]

    # Create HDF5 file
    with h5py.File(f'{current_dir}/robot_data.hdf5', 'w') as f:
        # Create main data group
        data_group = f.create_group('data')
        
        # Create datasets for each sample
        for i in range(num_samples):
            sample_group = data_group.create_group(f'sample_{i:05d}')
            
            # Store task description
            sample_group.create_dataset('task_description', 
                                      data=task_descriptions[i].encode('utf-8'))
            
            # Store task embedding
            sample_group.create_dataset('task_embedding', 
                                      data=task_embeddings[i])
            
            # Store images
            sample_group.create_dataset('rgb', 
                                      data=rgb_images[i],
                                      compression='gzip',
                                      compression_opts=9)
            
            sample_group.create_dataset('depth', 
                                      data=depth_images[i],
                                      compression='gzip',
                                      compression_opts=9)
            
            sample_group.create_dataset('object_mask', 
                                      data=object_masks[i],
                                      compression='gzip',
                                      compression_opts=9)
            
            # Store robot state
            sample_group.create_dataset('robot_joints', 
                                      data=robot_joints[i])
            
            sample_group.create_dataset('gripper_pose', 
                                      data=gripper_poses[i])
            
            # Add metadata
            sample_group.attrs['timestamp'] = str(np.datetime64('now')).encode('utf-8')
        
        # Add file-level attributes
        f.attrs['total_samples'] = num_samples
        f.attrs['created_at'] = str(np.datetime64('now')).encode('utf-8')
        f.attrs['description'] = 'Robot Task Dataset'.encode('utf-8')
        
        # Add format description
        format_desc = {
            'rgb': '480x640x3 uint8 RGB image',
            'depth': '480x640 float32 depth image',
            'object_mask': '480x640 uint8 object mask',
            'robot_joints': '7-dim joint angles',
            'gripper_pose': '7-dim pose [x,y,z,qx,qy,qz,qw]',
            'task_embedding': '512-dim task embedding vector'
        }
        # Encode all values to bytes
        encoded_desc = {k: v.encode('utf-8') for k, v in format_desc.items()}
        f.attrs['format_description'] = str(encoded_desc).encode('utf-8')

if __name__ == '__main__':
    create_and_write_hdf5()
    print("HDF5 data file created successfully!")
