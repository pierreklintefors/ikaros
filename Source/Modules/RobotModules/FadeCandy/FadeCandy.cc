//
//	FadeCandy.cc		This file is a part of the IKAROS project
//
//    Copyright (C) 2025 Birger Johansson
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

// Using fadecandy driver from https://github.com/iron-ox/fadecandy_ros/tree/master/fadecandy_driver

#include "fadecandy_driver.h" // Include the fadecandy driver

#include "ikaros.h"

using namespace ikaros;
using namespace fadecandy_driver;

class FadeCandy : public Module
{
    matrix RightEye, LeftEye, MouthLow, MouthHigh, Intensity;
    fadecandy_driver::FadecandyDriver fd_driver;
    parameter simulate;

    // Pre-allocated LED array colors for better performance
    std::vector<std::vector<Color>> led_array_colors;
    
    // Cache intensity values to avoid repeated matrix access
    float cached_left_intensity = 1.0f;
    float cached_right_intensity = 1.0f; 
    float cached_mouth_intensity = 1.0f;
    
    // Track if colors actually changed to avoid unnecessary updates
    bool colors_changed = false;

    void Init()
    {

        Bind(LeftEye, "LEFT_EYE");
        Bind(RightEye, "RIGHT_EYE");
        Bind(MouthHigh, "MOUTH_HIGH");
        Bind(MouthLow, "MOUTH_LOW");
        Bind(Intensity, "INTENSITY");

        Bind(simulate, "simulate");

        // Pre-allocate LED array with exact sizes for better performance
        led_array_colors.resize(4);
        led_array_colors[0].resize(8, Color(0, 0, 0));  // Mouth low
        led_array_colors[1].resize(8, Color(0, 0, 0));  // Mouth high  
        led_array_colors[2].resize(12, Color(0, 0, 0)); // Right eye
        led_array_colors[3].resize(12, Color(0, 0, 0)); // Left eye

        if (simulate)
        {
            Notify(msg_warning, "Simulate fadecandy");
            return;
        }
        if ((LeftEye.size_x() <= 12 && LeftEye.size_y() != 3))
            Notify(msg_warning, "Input LEFT_EYE size is not 3x12");
        if ((RightEye.size() <= 12 && RightEye.size_y() != 3))
            Notify(msg_warning, "Input RIGHT_EYE size is not 3x12");
        if ((MouthHigh.size() <= 8 && MouthHigh.size_y() != 3))
            Notify(msg_warning, "Input MOUTH_HIGH size is not 3x8");
        if ((MouthLow.size() <= 8 && MouthLow.size_y() != 3))
            Notify(msg_warning, "Input MOUTH_LOW size is not 3x8");

       

        try
        {
            auto serial_number = fd_driver.connect();
            Notify(msg_debug, "Connected to Fadecandy board: " + serial_number);
        }
        catch (const std::exception &e)
        {
            Notify(msg_warning, "Could not connect to Fadecandy board"); // Should this be fatal?
            return;
        }
    }

    void
    Tick()
    {
        if (simulate)
            return;

        if (!fd_driver.isConnected())
        {
            try
            {
                Notify(msg_debug, "Reconnecting to Fadecandy board: ");
                auto serial_number = fd_driver.connect();
                Notify(msg_warning, "Reconnecting to Fadecandy board: " + serial_number);
            }
            catch (const std::exception &e)
            {
                Notify(msg_debug, "Could not connect to Fadecandy board");
                return;
            }
        }   
            
        // Cache intensity values once per tick to avoid repeated matrix access
        cached_left_intensity = Intensity[0];
        cached_right_intensity = Intensity[1]; 
        cached_mouth_intensity = Intensity[2];
        
        colors_changed = false;

        // Fill color from input with optimized calculations
        for (size_t i = 0; i < 8; ++i) // 8 Leds in each row of the mouth
        {
            // Calculate colors once
            Color new_mouth_high(MouthHigh[0][i] * 255 * cached_mouth_intensity, 
                                 MouthHigh[1][i] * 255 * cached_mouth_intensity, 
                                 MouthHigh[2][i] * 255 * cached_mouth_intensity);
            Color new_mouth_low(MouthLow[0][i] * 255 * cached_mouth_intensity, 
                                MouthLow[1][i] * 255 * cached_mouth_intensity, 
                                MouthLow[2][i] * 255 * cached_mouth_intensity);
            
            // Only update if colors changed (compare RGB values manually)
            Color& current_mouth_high = led_array_colors[1][i];
            if (current_mouth_high.r_ != new_mouth_high.r_ || 
                current_mouth_high.g_ != new_mouth_high.g_ || 
                current_mouth_high.b_ != new_mouth_high.b_) {
                led_array_colors[1][i] = new_mouth_high;
                colors_changed = true;
            }
            
            Color& current_mouth_low = led_array_colors[0][i];
            if (current_mouth_low.r_ != new_mouth_low.r_ || 
                current_mouth_low.g_ != new_mouth_low.g_ || 
                current_mouth_low.b_ != new_mouth_low.b_) {
                led_array_colors[0][i] = new_mouth_low;
                colors_changed = true;
            }
        }
        for (size_t i = 0; i < 12; ++i) // 12 Leds in each eye
        {
            // Calculate colors once
            Color new_right_eye(RightEye[0][i] * 255 * cached_right_intensity, 
                               RightEye[1][i] * 255 * cached_right_intensity, 
                               RightEye[2][i] * 255 * cached_right_intensity);
            Color new_left_eye(LeftEye[0][i] * 255 * cached_left_intensity, 
                               LeftEye[1][i] * 255 * cached_left_intensity, 
                               LeftEye[2][i] * 255 * cached_left_intensity);
            
            // Only update if colors changed (compare RGB values manually)
            Color& current_right_eye = led_array_colors[2][i];
            if (current_right_eye.r_ != new_right_eye.r_ || 
                current_right_eye.g_ != new_right_eye.g_ || 
                current_right_eye.b_ != new_right_eye.b_) {
                led_array_colors[2][i] = new_right_eye;
                colors_changed = true;
            }
            
            Color& current_left_eye = led_array_colors[3][i];
            if (current_left_eye.r_ != new_left_eye.r_ || 
                current_left_eye.g_ != new_left_eye.g_ || 
                current_left_eye.b_ != new_left_eye.b_) {
                led_array_colors[3][i] = new_left_eye;
                colors_changed = true;
            }
        }

        // Only send colors if they actually changed
        if (colors_changed) {
            try
            {
                fd_driver.setColors(led_array_colors); // Send the colors to the driver. Set color is checking that isConnected is true. However,if the device is unpluged this is not detected by the driver.
            }
            catch (const std::exception &e)
            {
                Debug("Could not set colors of the eyes");
            }
        }
        
    
    }

    ~FadeCandy()
    {
        // Turn off all LEDs by setting them to black
        for (auto& row : led_array_colors) {
            for (auto& led : row) {
                led = Color(0, 0, 0);
            }
        }

        try {
            if (fd_driver.isConnected()) {
                fd_driver.setColors(led_array_colors);
            }
        }
        catch (const std::exception& e) {
            Debug("Error shutting off LEDs during destruction: " + std::string(e.what()));
        }
    }
    
    
};

INSTALL_CLASS(FadeCandy)
