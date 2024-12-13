is_patched = False


def patch_micro_particle_system():
    from omnigibson.systems.micro_particle_system import MicroPhysicalParticleSystem

    if hasattr(MicroPhysicalParticleSystem, '_original_sync_particle_instancers'):
        return

    MicroPhysicalParticleSystem._original_sync_particle_instancers = MicroPhysicalParticleSystem._sync_particle_instancers

    def new_sync_particle_instancers(self, idns, particle_groups, particle_counts):
        idns = [int(idn) for idn in idns]
        particle_groups = [int(group) for group in particle_groups]
        particle_counts = [int(count) for count in particle_counts]
        return self._original_sync_particle_instancers(idns, particle_groups, particle_counts)

    MicroPhysicalParticleSystem._sync_particle_instancers = new_sync_particle_instancers



def patch_vision_utils():
    from omnigibson.utils.vision_utils import Remapper
    
    if hasattr(Remapper, '_original_remap'):
        return
        
    Remapper._original_remap = Remapper.remap
    
    def new_remap(self, old_mapping, new_mapping, image, image_keys=None):
        if image.numel() == 0:
            return image, new_mapping
        return self._original_remap(old_mapping, new_mapping, image, image_keys)
    
    Remapper.remap = new_remap

def apply_all_patches():
    global is_patched
    if is_patched:
        return
    is_patched = True
    patch_micro_particle_system()
    patch_vision_utils()
    print("omnigibson is patched by btgym")