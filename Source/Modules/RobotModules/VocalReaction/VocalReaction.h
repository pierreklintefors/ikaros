//
//	VocalReaction.h		This file is a part of the IKAROS project
//
//    Copyright (C) 2012 <Author Name>
//
//    This program is free software; you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation; either version 2 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program; if not, write to the Free Software
//    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
//    See http://www.ikaros-project.org/ for more information.
//

#ifndef VocalReaction_
#define VocalReaction_

#include "IKAROS.h"

class VocalReaction: public Module
{
public:
    static Module * Create(Parameter * p) { return new VocalReaction(p); }

    VocalReaction(Parameter * p) : Module(p) {}
    virtual ~VocalReaction();

    void 		Init();
    void 		Tick();

    //Randomizer function
    int     Random(int,int);

    bool        pause;
    bool        bored_played;
    Timer       timer;
    // pointers to inputs and outputs
    // and integers to represent their sizes

    float *     input_array;
    int         input_array_size;

    float **    input_Pos_matrix;
    int         input_Pos_matrix_size_x;
    int         input_Pos_matrix_size_y;

    float **    input_ID_matrix;
    int         input_ID_matrix_size_x;
    int         input_ID_matrix_size_y;

    float *     reaction_output_array;
    int         reaction_output_array_size;

    float **    output_matrix;
    int         output_matrix_size_x;
    int         output_matrix_size_y;

    // internal data storage

    float *     internal_array;
    float **    internal_matrix;

    // parameter values

    int         num_categories;
    int         num_sounds_per_category;
    int         num_intensity_levels;
    float       pauseTime;
    int         valid_repetitions;
    bool        earcons;

    //
    int         input_repetition;
    int         object_id_previous_tick;


};

#endif
