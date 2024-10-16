#!/usr/bin/env python3
import os

os.system('gmsh -2 -format msh2 ../mesh/2dMeshFine.geo -o ../mesh/new_mesh.msh')