# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for The Machine HUD Simulator — one-folder build."""

import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

block_cipher = None

# Collect face_recognition_models data files (shape predictors, CNN models)
fr_models_datas = collect_data_files('face_recognition_models')
dlib_binaries = collect_dynamic_libs('dlib')

a = Analysis(
    ['machine_vision.py'],
    pathex=[],
    binaries=dlib_binaries,
    datas=fr_models_datas + [
        ('known_faces/.gitkeep', 'known_faces'),
    ],
    hiddenimports=[
        'face_recognition',
        'face_recognition_models',
        'dlib',
        'cv2',
        'numpy',
        'numpy.core._methods',
        'numpy.lib.format',
        'PIL',
        'PIL.Image',
        'json',
        'pickle',
        'pkg_resources',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'mediapipe'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='TheMachine',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TheMachine',
)
