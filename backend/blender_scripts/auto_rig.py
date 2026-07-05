"""
Runs INSIDE Blender (--background --python auto_rig.py -- --input ... --output ...).

Self-hosted replacement for Mixamo's auto-rigger (see rigging.py's module
docstring for why). No network calls, no GPU — pure mesh geometry + Blender's
own armature/skinning tools.

ASSUMPTIONS (tune the constants below if a specific backend violates them):
  - After import, Blender's own importer axis-conversion has already put the
    character upright along Z, facing roughly +/-Y, arms spread along X.
    This is the default behavior of Blender's glTF/FBX/OBJ importers for
    content authored the conventional way (which TripoSR/CharacterGen/
    InstantMesh/TRELLIS all are) — if one backend's exporter turns out to
    violate it, override with axis_forward/axis_up on that importer call.
  - Character stands with feet at the mesh's minimum Z and head at maximum Z.
  - Reference pose is A-pose or T-pose (arms held away from torso) — true for
    CharacterGen's canonical output, and the reference images already say
    "T-pose arms extended straight out" in the filename. If a mesh comes in
    arms-down, the arm-placement heuristic falls back to fixed proportions
    (flagged in the report as used_fallback_arm_pose) rather than guessing.

Writes a JSON "report" sidecar on both success and failure so the calling
process (rigging.py) never has to re-parse the binary FBX — Blender already
has the bone graph in memory, so the hierarchy check happens right here.
"""

import sys
import json
import argparse
import traceback

import bpy


# ---- tunable constants -----------------------------------------------------

MIN_CLUSTER_VERTS = 8          # min verts per side to trust a 2-cluster split
GAP_FRACTION_THRESHOLD = 0.12  # gap must exceed this fraction of body width
LEG_SCAN_RANGE = (0.30, 0.62)  # fraction of height, scanned top->down
LEG_SCAN_STEPS = 48
SHOULDER_SCAN_RANGE = (0.60, 0.97)
SHOULDER_SCAN_STEPS = 48
SLICE_THICKNESS_FRAC = 0.012   # fraction of height per horizontal slice
MIN_BONE_LENGTH = 0.01         # meters; guards against Blender auto-dropping
                                # zero-length bones


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--pose", default="unknown")
    return p.parse_args(argv)


def write_report(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for coll in list(bpy.data.collections):
        if coll.name != "Collection" and coll.users == 0:
            bpy.data.collections.remove(coll)


def import_mesh(path):
    suffix = path.lower().rsplit(".", 1)[-1]
    if suffix in ("glb", "gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif suffix == "fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif suffix == "obj":
        try:
            bpy.ops.wm.obj_import(filepath=path)  # Blender 4.0+
        except AttributeError:
            bpy.ops.import_scene.obj(filepath=path)  # Blender <4.0
    else:
        raise ValueError(f"Unsupported mesh format: .{suffix}")

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise ValueError(f"Import produced no mesh objects from {path}")

    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()

    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def world_bbox(obj):
    # Vertex .co is already in world space here since transform_apply()
    # baked location/rotation/scale into the mesh data on import.
    xs = [v.co.x for v in obj.data.vertices]
    ys = [v.co.y for v in obj.data.vertices]
    zs = [v.co.z for v in obj.data.vertices]
    return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))


def slice_x_clusters(obj, z_center, thickness):
    """Vertices whose Z falls in [z_center - t/2, z_center + t/2], split into
    left/right clusters by the largest gap along X. Returns None if no clean
    2-cluster split is found."""
    lo, hi = z_center - thickness / 2, z_center + thickness / 2
    xs = sorted(v.co.x for v in obj.data.vertices if lo <= v.co.z <= hi)
    if len(xs) < 2 * MIN_CLUSTER_VERTS:
        return None

    body_width = xs[-1] - xs[0]
    if body_width <= 1e-6:
        return None

    best_gap, best_idx = 0.0, None
    for i in range(1, len(xs)):
        gap = xs[i] - xs[i - 1]
        if gap > best_gap:
            best_gap, best_idx = gap, i

    if best_idx is None or best_gap < GAP_FRACTION_THRESHOLD * body_width:
        return None

    left, right = xs[:best_idx], xs[best_idx:]
    if len(left) < MIN_CLUSTER_VERTS or len(right) < MIN_CLUSTER_VERTS:
        return None

    left_center = sum(left) / len(left)
    right_center = sum(right) / len(right)
    return {"left_x": left_center, "right_x": right_center, "gap": best_gap}


def find_leg_split(obj, z_min, height):
    thickness = SLICE_THICKNESS_FRAC * height
    lo_frac, hi_frac = LEG_SCAN_RANGE
    # scan top -> down; first successful split (highest Z) is the crotch
    for i in range(LEG_SCAN_STEPS):
        frac = hi_frac - (hi_frac - lo_frac) * (i / (LEG_SCAN_STEPS - 1))
        z = z_min + frac * height
        result = slice_x_clusters(obj, z, thickness)
        if result:
            return {
                "z": z,
                "left_x": result["left_x"],
                "right_x": result["right_x"],
                "fallback": False,
            }
    # Fallback: dress/saree geometry (or any single-blob lower body) never
    # resolves into two clusters. Use a symmetric proportional estimate
    # instead of failing the whole rig over it.
    center_x = sum(v.co.x for v in obj.data.vertices) / len(obj.data.vertices)
    approx_half_width = 0.045 * height
    return {
        "z": z_min + 0.5 * height,
        "left_x": center_x - approx_half_width,
        "right_x": center_x + approx_half_width,
        "fallback": True,
    }


def find_shoulders_and_hands(obj, z_min, height, torso_center_x):
    thickness = SLICE_THICKNESS_FRAC * height
    lo_frac, hi_frac = SHOULDER_SCAN_RANGE
    best = None
    for i in range(SHOULDER_SCAN_STEPS):
        frac = hi_frac - (hi_frac - lo_frac) * (i / (SHOULDER_SCAN_STEPS - 1))
        z = z_min + frac * height
        lo, hi = z - thickness / 2, z + thickness / 2
        xs = [v.co.x for v in obj.data.vertices if lo <= v.co.z <= hi]
        if len(xs) < 2 * MIN_CLUSTER_VERTS:
            continue
        extent = max(xs) - min(xs)
        if best is None or extent > best["extent"]:
            best = {"z": z, "extent": extent, "min_x": min(xs), "max_x": max(xs)}

    torso_width_guess = 0.20 * height  # rough shoulder-width-only baseline
    if best is None or best["extent"] < torso_width_guess * 1.3:
        # Arms aren't spread wide of the torso in this pose (or detection
        # failed) — use fixed A-pose-ish proportions instead of guessing
        # from noisy geometry.
        shoulder_z = z_min + 0.82 * height
        half_shoulder = 0.11 * height
        hand_half = 0.34 * height
        return {
            "shoulder_z": shoulder_z,
            "left_shoulder_x": torso_center_x - half_shoulder,
            "right_shoulder_x": torso_center_x + half_shoulder,
            "left_hand_x": torso_center_x - hand_half,
            "right_hand_x": torso_center_x + hand_half,
            "fallback": True,
        }

    half_shoulder = 0.11 * height
    return {
        "shoulder_z": best["z"],
        "left_shoulder_x": torso_center_x - half_shoulder,
        "right_shoulder_x": torso_center_x + half_shoulder,
        "left_hand_x": best["min_x"],
        "right_hand_x": best["max_x"],
        "fallback": False,
    }


def add_bone(eb, name, head, tail, parent=None):
    b = eb.new(name)
    b.head = head
    b.tail = tail
    if (b.tail - b.head).length < MIN_BONE_LENGTH:
        b.tail = (b.head[0], b.head[1], b.head[2] + MIN_BONE_LENGTH)
    if parent is not None:
        b.parent = parent
        b.use_connect = False
    return b


def build_armature(mesh_obj, z_min, z_max, leg, arms):
    height = z_max - z_min
    hips_z = leg["z"]
    chest_z = arms["shoulder_z"]
    spine_z = hips_z + 0.35 * (chest_z - hips_z)
    neck_z = chest_z + 0.06 * height
    head_tip_z = z_max
    knee_z = z_min + 0.5 * hips_z if hips_z > z_min else z_min + 0.25 * height
    ankle_z = z_min + 0.04 * height
    toe_z = z_min

    center_x = (leg["left_x"] + leg["right_x"]) / 2.0
    hip_l_x, hip_r_x = leg["left_x"], leg["right_x"]

    arm_data = bpy.data.armatures.new("CharacterArmature")
    arm_obj = bpy.data.objects.new("CharacterArmature", arm_data)
    bpy.context.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_data.edit_bones

    root = add_bone(eb, "root", (center_x, 0, z_min), (center_x, 0, hips_z))
    hips = add_bone(eb, "hips", (center_x, 0, hips_z), (center_x, 0, spine_z), root)
    spine = add_bone(eb, "spine", (center_x, 0, spine_z), (center_x, 0, chest_z), hips)
    chest = add_bone(eb, "chest", (center_x, 0, chest_z), (center_x, 0, neck_z), spine)
    neck = add_bone(eb, "neck", (center_x, 0, neck_z), (center_x, 0, neck_z + 0.03 * height), chest)
    add_bone(eb, "head", (center_x, 0, neck_z + 0.03 * height), (center_x, 0, head_tip_z), neck)

    for side, sign in (("L", -1), ("R", 1)):
        sh_x = arms["left_shoulder_x"] if sign < 0 else arms["right_shoulder_x"]
        hand_x = arms["left_hand_x"] if sign < 0 else arms["right_hand_x"]
        elbow_x = sh_x + (hand_x - sh_x) * 0.5
        shoulder = add_bone(eb, f"shoulder.{side}", (center_x, 0, chest_z), (sh_x, 0, chest_z), chest)
        upperarm = add_bone(eb, f"upperarm.{side}", (sh_x, 0, chest_z), (elbow_x, 0, chest_z), shoulder)
        lowerarm = add_bone(eb, f"lowerarm.{side}", (elbow_x, 0, chest_z), (hand_x, 0, chest_z), upperarm)
        add_bone(eb, f"hand.{side}", (hand_x, 0, chest_z), (hand_x + sign * 0.05 * height, 0, chest_z), lowerarm)

        hip_x = hip_l_x if sign < 0 else hip_r_x
        upperleg = add_bone(eb, f"upperleg.{side}", (hip_x, 0, hips_z), (hip_x, 0, knee_z), hips)
        lowerleg = add_bone(eb, f"lowerleg.{side}", (hip_x, 0, knee_z), (hip_x, 0, ankle_z), upperleg)
        add_bone(eb, f"foot.{side}", (hip_x, 0, ankle_z), (hip_x, 0.08 * height, toe_z), lowerleg)

    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj


def validate_hierarchy(arm_obj):
    bones = arm_obj.data.bones
    names = {b.name for b in bones}

    def chain_ok(start, expected_len_min):
        chain = []
        b = bones.get(start)
        depth = 0
        while b is not None and depth < 10:
            chain.append(b.name)
            children = [c for c in bones if c.parent == b]
            if not children:
                break
            b = children[0]
            depth += 1
        return len(chain) >= expected_len_min, chain

    main_ok, main_chain = chain_ok("root", 5)  # root,hips,spine,chest,neck/head
    arms_ok = all(f"hand.{s}" in names and f"upperarm.{s}" in names for s in ("L", "R"))
    legs_ok = all(f"foot.{s}" in names and f"upperleg.{s}" in names for s in ("L", "R"))

    parents_sane = all(
        bones[f"upperarm.{s}"].parent and bones[f"upperarm.{s}"].parent.name == f"shoulder.{s}"
        for s in ("L", "R")
    ) and all(
        bones[f"upperleg.{s}"].parent and bones[f"upperleg.{s}"].parent.name == "hips"
        for s in ("L", "R")
    )

    return bool(main_ok and arms_ok and legs_ok and parents_sane), main_chain


def skin_mesh(mesh_obj, arm_obj):
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")


def main():
    args = parse_args()
    report = {
        "bone_count": 0,
        "has_standard_hierarchy": False,
        "bone_names": [],
        "used_fallback_leg_split": False,
        "used_fallback_arm_pose": False,
        "pose_hint": args.pose,
        "error": None,
    }
    try:
        clear_scene()
        mesh_obj = import_mesh(args.input)
        (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = world_bbox(mesh_obj)
        height = z_hi - z_lo
        if height <= 1e-6 or len(mesh_obj.data.vertices) < 50:
            raise ValueError(
                f"Mesh at {args.input} looks empty/degenerate "
                f"(verts={len(mesh_obj.data.vertices)}, height={height}) — "
                f"nothing to rig."
            )

        leg = find_leg_split(mesh_obj, z_lo, height)
        torso_center_x = (leg["left_x"] + leg["right_x"]) / 2.0
        arms = find_shoulders_and_hands(mesh_obj, z_lo, height, torso_center_x)

        arm_obj = build_armature(mesh_obj, z_lo, z_hi, leg, arms)
        skin_mesh(mesh_obj, arm_obj)

        has_standard, main_chain = validate_hierarchy(arm_obj)

        bpy.ops.object.select_all(action="DESELECT")
        mesh_obj.select_set(True)
        arm_obj.select_set(True)
        bpy.context.view_layer.objects.active = arm_obj
        bpy.ops.export_scene.fbx(
            filepath=args.output,
            use_selection=True,
            add_leaf_bones=False,
            bake_anim=False,
        )

        # FIX (2026-07-05): GLB export was silently dropping the glTF `skin`
        # binding (validator: NODE_SKINNED_MESH_WITHOUT_SKIN). Vertex
        # JOINTS_0/WEIGHTS_0 + a full valid bone hierarchy existed, but no
        # glTF `skin` object tied them together. FBX export above is
        # confirmed fine (Deformer/SubDeformer/Cluster/BindPose all present),
        # so this was GLB-exporter-specific. Fix: re-select mesh+armature
        # right before the glTF export call (in case FBX export altered
        # selection state) and pass export flags explicitly.
        bpy.ops.object.select_all(action="DESELECT")
        mesh_obj.select_set(True)
        arm_obj.select_set(True)
        bpy.context.view_layer.objects.active = arm_obj

        glb_path = args.output.rsplit(".", 1)[0] + ".glb"
        bpy.ops.export_scene.gltf(
            filepath=glb_path,
            export_format="GLB",
            use_selection=True,
            export_skins=True,
            export_apply=False,
            export_yup=True,
        )

        report.update({
            "bone_count": len(arm_obj.data.bones),
            "has_standard_hierarchy": has_standard,
            "bone_names": [b.name for b in arm_obj.data.bones],
            "used_fallback_leg_split": leg["fallback"],
            "used_fallback_arm_pose": arms["fallback"],
            "main_chain": main_chain,
        })
        write_report(args.report, report)

    except Exception as e:
        report["error"] = f"{e}\n{traceback.format_exc()}"
        write_report(args.report, report)
        sys.exit(1)


if __name__ == "__main__":
    main()
