/*
 * GPU Smoldyn: Smoldyn algorithm ported to the GPU using CUDA 2.2
 * Writtern By Lorenzo Dematté, 2010-2011
 *
 * This file is part of GPU Smoldyn
 * 
 *     GPU Smoldyn is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 * 
 *     GPU Smoldyn is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with Foobar.  If not, see <http://www.gnu.org/licenses/>. 
 * 
 * Based on algorithm and source code of Smoldyn, written by Steve Andrews, 2003.
 * 
 * Portions taken by code examples in NVIDIA Whitepapers, GPU Gems 2 and 3, 
 * Copyright 1993-2009 NVIDIA Corporation, Addison-Wesley and the original authors. 
 * 
 */

#ifndef __RENDER_PARTICLES__
#define __RENDER_PARTICLES__

class ParticleRenderer
{
public:
    ParticleRenderer();
    ~ParticleRenderer();

    void setPositions(float *pos, int numParticles);
    void setBufferObjects(unsigned int posVBO, unsigned int typesVBO, int numParticles);
    //void setTypesBuffer(unsigned int vbo) { m_typesVBO = vbo; }

    enum DisplayMode
    {
        PARTICLE_POINTS,
        PARTICLE_SPHERES,
        PARTICLE_NUM_MODES
    };

    void display(DisplayMode mode = PARTICLE_POINTS);
    void displayGrid();

    void setPointSize(float size)  { m_pointSize = size; }
    void setParticleRadius(float r) { m_particleRadius = r; }
    void setFOV(float fov) { m_fov = fov; }
    void setWindowSize(int w, int h) { m_window_w = w; m_window_h = h; }

protected: // methods
    void _initGL();
    void _drawPoints();
    GLuint _compileProgram(const char *vsource, const char *fsource);

protected: // data
    float *m_pos;
    int m_numParticles;

    float m_pointSize;
    float m_particleRadius;
    float m_fov;
    int m_window_w, m_window_h;

    GLuint m_program;

    GLuint m_vbo;
    GLuint m_typesVBO;
};

#endif //__ RENDER_PARTICLES__
