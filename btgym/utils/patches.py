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


def patch_cloth_prim():
    from omnigibson.prims.cloth_prim import ClothPrim

    def set_particle_positions(self, positions, idxs=None):
        """
        Sets individual particle positions for this cloth prim

        Args:
            positions (n-array): (N, 3) numpy array, where each of the N particles' positions are expressed in (x,y,z)
                cartesian coordinates relative to the world frame
            idxs (n-array or None): If set, will only set the requested indexed particle state
        """
        n_expected = self._n_particles if idxs is None else len(idxs)
        # assert (
        #     len(positions) <= n_expected
        # ), f"Got mismatch in particle setting size: {len(positions)}, vs. number of expected particles {n_expected}!"

        if len(positions) < n_expected:
            positions = th.cat([positions, th.repeat(positions[-1], n_expected - len(positions), dim=0)], dim=0)
        elif len(positions) > n_expected:
            positions = positions[:n_expected]

        translation, rotation = self.get_position_orientation()
        rotation = T.quat2mat(rotation)
        scale = self.scale
        p_local = (rotation.T @ (positions - translation).T).T / scale

        # Fill the idxs if requested
        if idxs is not None:
            p_local_old = vtarray_to_torch(self.get_attribute(attr="points"))
            p_local_old[idxs] = p_local
            p_local = p_local_old

        self.set_attribute(attr="points", val=lazy.pxr.Vt.Vec3fArray(p_local.tolist()))

    ClothPrim.set_particle_positions = ClothPrim.set_particle_positions

def apply_all_patches():
    global is_patched
    if is_patched:
        return
    is_patched = True
    patch_micro_particle_system()
    patch_vision_utils()
    print("omnigibson is patched by btgym")