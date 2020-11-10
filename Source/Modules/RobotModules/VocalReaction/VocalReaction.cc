//
//	VocalReaction.cc		This file is a part of the IKAROS project
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
//  This example is intended as a starting point for writing new Ikaros modules
//  The example includes most of the calls that you may want to use in a module.
//  If you prefer to start with a clean example, use he module MinimalModule instead.
//

#include "VocalReaction.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include "IKAROS.h"

using namespace std;

// use the ikaros namespace to access the math library
// this is preferred to using <cmath>

using namespace ikaros;


void
VocalReaction::Init()
{
    // The parameters are initialized
    // from the IKC and can optionally be changed from the
    // user interface while Ikaros is running. If the parameter is not
    // set, the default value specified in the ikc-file will be used instead.

    //Sets value of int varibles num_categories number_sound_categories, num_intensity_levels according to values of parameters inputs in user interface
    Bind(num_categories, "number_sound_categories");
    Bind (num_sounds_per_category, "sounds_per_category");
    Bind(num_intensity_levels, "intensity_levels");
    Bind(pauseTime, "pause_in_miliseconds");
    Bind(valid_repetitions, "repetitions_before_bored");
    Bind(earcons, "earcons");


    io(input_Pos_matrix, input_Pos_matrix_size_x, input_Pos_matrix_size_y, "POSITION_INPUT");

    // Get pointer to a matrix and treat it as a matrix. If an array is
    // connected to this input, size_y will be 1.

    io(input_ID_matrix, input_ID_matrix_size_x, input_ID_matrix_size_y, "OBJECT_INPUT");

    // Do the same for the outputs

    io(reaction_output_array, reaction_output_array_size, "REACTION_OUTPUT");


    // Allocate some data structures to use internaly
    // in the module

    // Create an array with elements corresponding to number of sounds in the soundlibrary
    // To access the array use internal_array[i].

    internal_array = create_array(num_categories * num_sounds_per_category * num_intensity_levels);

    // Create a matrix with the same size as POSITION_INPUT
    // IMPORTANT: For the matrix the sizes are given in order X, Y
    // in all functions including when the matrix is created
    // See: http://www.ikaros-project.org/articles/2007/datastructures/

    //Bool changed to true if object has been identified an tick should pause
    bool pause = false;

    //Bool set to true if same input is repeated over the valid limit of repetitions and a bored sound has been played
    bool bored_played = false;

    //Create a timer constructor
    Timer timer;

    //Counter that increases when current input is the same as previous
    input_repetition = 0;

    object_id_previous_tick = 0;

    //Boundaries for randomization function



}

VocalReaction::~VocalReaction()
{
    // In general, a destructor is only necessary
    // when a module communicates with external devices etc
    // All modules are destroyed when Ikaros stops

    // Destroy data structures that you allocated in Init.

    destroy_array(internal_array);
    destroy_matrix(internal_matrix);

    // Do NOT destroy data structures that you got from the
    // kernel with io or GetInputArray, GetInputMatrix etc.
}

//Function that generates a random number between a specific range
int
VocalReaction::Random(int min, int max) //range : [min, max]
{
   static bool first = true;
   if (first)
   {
      float seed = std::time(NULL);
      srand(seed); //seeding for the first time only!
      first = false;
   }
   return min + rand() % (( max + 1 ) - min);
}




void
VocalReaction::Tick()
{

  //Sets all elements in reaction_output_array to 0
  for (int i = 0; i < reaction_output_array_size; i++)
  {
    reaction_output_array[i] = 0;
  }

  bool no_output = false;
  
  bool bored = false;
  
  int lower_bound;
  int upper_bound;
    

  //Checks if enough time has been gone by before enabling a new object to be identified   
  if(pause == false || timer.GetTime() > pauseTime)
  {
    //Search the input matrix to see if a marker has been identified
    for (int j = 0; j < input_ID_matrix_size_y; j++)
    {  for (int i = 0; i < input_ID_matrix_size_x; i++)
      {
        if (input_ID_matrix[j][i] > 0)
        {

         
          //Starts timer
          timer.Restart();

          pause = true;

          //If the object id is 1 - the question is " are you ready?" Which can two answeres (double span size)
          if(input_ID_matrix[j][i]==1)
          {
            lower_bound = input_ID_matrix[j][i];
            upper_bound = (num_sounds_per_category*num_intensity_levels) *2;
          }

          else if(input_ID_matrix[j][i]==2)
          {
            lower_bound = (num_sounds_per_category*num_intensity_levels)*2;
            upper_bound = lower_bound + num_sounds_per_category*num_intensity_levels;
          }
          else
          {
            //Sets lower and higher boundary for randomizing a sound in the identified category
            lower_bound = input_ID_matrix[j][i] * num_sounds_per_category* num_intensity_levels;
            upper_bound = lower_bound + num_sounds_per_category*num_intensity_levels;
          }
          


          if(earcons)
          {
            
          
            //If the object id is 1 - the question is " are you ready?" Which can two answeres (double span size)
            if(input_ID_matrix[j][i]==1)
            {
              lower_bound = input_ID_matrix[j][i]-1;
              upper_bound = (lower_bound + num_sounds_per_category * num_intensity_levels);
            }
            else if(input_ID_matrix[j][i]==2)
            {
              lower_bound = input_ID_matrix[j][i]+(num_sounds_per_category*num_intensity_levels)-1;
              upper_bound = lower_bound;
            }
            else
            {
              //Sets lower and higher boundary for randomizing a sound in the identified category
              lower_bound = input_ID_matrix[j][i] ;
              upper_bound = input_ID_matrix[j][i] ;
            }
            

          } 
          //Checks if the scanned object is the same as previous.
          if(object_id_previous_tick == input_ID_matrix[j][i])  
          {
            input_repetition +=1;
            no_output = true;
          }
          //If the th eobject is not the same as prevoius one, repetition count is set to 0 and n 
          else
          { 
            input_repetition = 0;
            bored_played = false;   
          }
          //Checks if same object has been repeated to the point of set boundary of valid repetitions 
          if(input_repetition > valid_repetitions && !bored_played && input_ID_matrix[j][i] !=1 )  
          {
            //These bounds corresponds to the bored audio files
            lower_bound = reaction_output_array_size - (num_sounds_per_category*num_intensity_levels);
            upper_bound = reaction_output_array_size;

            bored = true;
            

            cout << '\n';
            cout << "Input reset and play bored" << '\n';

          
          }     
          if(!earcons)
            {//Subtract 1 from bounds because index in SoundOutput input vector starts at 0
            lower_bound --; 
            upper_bound --;
            }
          //Generate a random number between boundaries
          int rand_num = Random(lower_bound, upper_bound);
          
          object_id_previous_tick = input_ID_matrix[j][i];

          //Sets the elemetn as 1 only if the object is not the same as previous or reached the limit of boredom. If prevoius played sound was bored, no output will be generated
          if(!no_output && !bored_played || bored )
          {
            //The index of the array in SoundOutput starts at 0
            reaction_output_array[rand_num] = 1;
            
            
            if(bored)
              bored_played = true;
          }
          
          

          //Prints some information about scanned object repetition
          cout << '\n';
          cout << "Input repetition: ";
          cout << input_repetition << '\n';
          
          std::cout << '\n';
          std::cout << "Object detected!"<< '\n';
          cout << "Lower: ";
          cout << lower_bound<< '\n';
          cout << "Upper: ";
          cout << upper_bound<< '\n';
          cout << "Element(starting from 0): ";
          cout << rand_num << '\n';
          std::cout << '\n';
          

          
          
          

        }
      }//For loop row
    }// For loop column

  }//If (timer)
  
       
//When enough time has passed without identification of an marker, the same marker id can be presented again
  if(timer.GetTime() > pauseTime )  
  {
    object_id_previous_tick = 0;
    no_output = false;
    std::cout << '\n';
    std::cout << "Enough time has passed for same object to be shown again "<< '\n';
    
  }
  

}// Tick()



// Install the module. This code is executed during start-up.

static InitClass init("VocalReaction", &VocalReaction::Create, "Source/Modules/RobotModules/VocalReaction/");
