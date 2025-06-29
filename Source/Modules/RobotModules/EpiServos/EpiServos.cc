//
//	EpiServos.cc		This file is a part of the IKAROS project
//
//    Copyright (C) 2022 Birger Johansson

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

// Dynamixel settings
#define PROTOCOL_VERSION 2.0 // See which protocol version is used in the Dynamixel
#define BAUDRATE1M 1000000   // XL-320 is limited to 1Mbit
#define BAUDRATE3M 3000000   // MX servos

#define ADDR_OPERATING_MODE 11
// Indirect adress
#define IND_ADDR_TORQUE_ENABLE 168
#define ADDR_TORQUE_ENABLE 64
#define IND_ADDR_GOAL_POSITION 170
#define ADDR_GOAL_POSITION 116
#define IND_ADDR_GOAL_CURRENT 178
#define ADDR_GOAL_CURRENT 102
#define IND_ADDR_GOAL_PWM 182
#define ADDR_GOAL_PWM 100

#define IND_ADDR_PRESENT_POSITION 578
#define ADDR_PRESENT_POSITION 132
#define IND_ADDR_PRESENT_CURRENT 586
#define ADDR_PRESENT_CURRENT 126
#define IND_ADDR_PRESENT_TEMPERATURE 590
#define ADDR_PRESENT_TEMPERATURE 146

// Common for the 2.0 (not XL320)
#define ADDR_PROFILE_ACCELERATION 108
#define ADDR_PROFILE_VELOCITY 112
#define ADDR_P 84
#define ADDR_I 82
#define ADDR_D 80

// ID of each dynamixel chain.
#define HEAD_ID_MIN 2
#define HEAD_ID_MAX 5

#define ARM_ID_MIN 2
#define ARM_ID_MAX 7

#define BODY_ID_MIN 2
#define BODY_ID_MAX 2

#define PUPIL_ID_MIN 2
#define PUPIL_ID_MAX 3

#define EPI_TORSO_NR_SERVOS 6
#define EPI_NR_SERVOS 19

#define TIMER_POWER_ON 2           // Timer ramping up sec
#define TIMER_POWER_OFF 5          // Timer ramping down sec
#define TIMER_POWER_OFF_EXTENDED 3 // Timer until torque enable off sec

#define HEAD_INDEX_IO 0
#define PUPIL_INDEX_IO 4
#define LEFT_ARM_INDEX_IO 6
#define RIGHT_ARM_INDEX_IO 12
#define BODY_INDEX_IO 18

#define MAX_TEMPERATURE 65

// Conversion factors (mA / Dynamixel unit) for MX-106 (2.0)
#define CURRENT_UNIT 3.36 // Use the same unit for Goal and Present Current

#define ADDR_MIN_POSITION_LIMIT 52
#define ADDR_MAX_POSITION_LIMIT 48

#define INDIRECTADDRESS_FOR_WRITE      168                  
#define INDIRECTADDRESS_FOR_READ       578                  
#define INDIRECTDATA_FOR_WRITE         224
#define INDIRECTDATA_FOR_READ          634



// TODO:
// Add fast sync write feature

#include <stdio.h>
#include <vector> // Data from dynamixel sdk
#include <future> // Threads

#include "ikaros.h"

// This must be after ikaros.h
#include "dynamixel_sdk.h" // Uses Dynamixel SDK library

#include "json.hpp"
#include <string>
#include <algorithm>


using json = nlohmann::json;

using namespace ikaros;
typedef struct
{
    std::string serialPortPupil;
    std::string serialPortHead;
    std::string serialPortBody;
    std::string serialPortLeftArm;
    std::string serialPortRightArm;
    std::string type;

} Robot_parameters;

class EpiServos : public Module
{
    // Paramteters
    parameter simulate;
    matrix minLimitPosition;
    matrix maxLimitPosition;
    parameter ServoControlMode;
    parameter dataToWrite;

    // Ikaros IO
    matrix goalPosition;
    matrix goalCurrent;
    matrix torqueEnable = true;
    matrix goalPWM;

    matrix presentPosition;
    matrix presentCurrent;

    bool EpiTorsoMode = false;
    bool EpiFullMode = false;

    int AngleMinLimitPupil[2];
    int AngleMaxLimitPupil[2];

    int len_write_data;

    std::string controlMode;


    dynamixel::PortHandler *portHandlerHead;
    dynamixel::PacketHandler *packetHandlerHead;
    dynamixel::GroupSyncRead *groupSyncReadHead;
    dynamixel::GroupSyncWrite *groupSyncWriteHead;

    dynamixel::PortHandler *portHandlerPupil;
    dynamixel::PacketHandler *packetHandlerPupil;
    // dynamixel::GroupSyncRead *groupSyncReadPupil;
    // dynamixel::GroupSyncWrite *groupSyncWritePupil;

    dynamixel::PortHandler *portHandlerLeftArm;
    dynamixel::PacketHandler *packetHandlerLeftArm;
    dynamixel::GroupSyncRead *groupSyncReadLeftArm;
    dynamixel::GroupSyncWrite *groupSyncWriteLeftArm;

    dynamixel::PortHandler *portHandlerRightArm;
    dynamixel::PacketHandler *packetHandlerRightArm;
    dynamixel::GroupSyncRead *groupSyncReadRightArm;
    dynamixel::GroupSyncWrite *groupSyncWriteRightArm;

    dynamixel::PortHandler *portHandlerBody;
    dynamixel::PacketHandler *packetHandlerBody;
    dynamixel::GroupSyncRead *groupSyncReadBody;
    dynamixel::GroupSyncWrite *groupSyncWriteBody;

    // Vectors for iteration
    std::vector<dynamixel::PortHandler*> portHandlers;
    std::vector<dynamixel::PacketHandler*> packetHandlers;

    std::string robotName;
    std::map<std::string, Robot_parameters> robot;

    matrix headData;
    matrix bodyData;
    matrix leftArmData;
    matrix rightArmData;
    matrix servoParameters;
    dictionary servoControlTable;
    list parameter_lst;

    int CalculateLenWriteData(std::string dataToWrite){
        int len = 1; // Torque enable
        std::stringstream ss(dataToWrite);
        std::string item;
        
        // Split string by comma and iterate
        while (std::getline(ss, item, ',')) {
            // Trim whitespace from item
            item.erase(0, item.find_first_not_of(" \t\r\n"));
            item.erase(item.find_last_not_of(" \t\r\n") + 1);
            
            Debug("Calculating length: Data to write: " + item);
            if (!item.empty()) {
                len += servoControlTable[item]["Bytes"];
                Debug("Calculating length: Bytes: " + std::to_string(servoControlTable[item]["Bytes"]));
            }
        }
        return len;
    }
    

    bool CommunicationPupil()
    {
        // Change this function.
        // No need to have torque enable.
        // Only goal position and use sync write.

        int index = 0;
        int dxl_comm_result = COMM_TX_FAIL; // Communication result
        bool dxl_addparam_result = false;   // addParam result
        bool dxl_getdata_result = false;    // GetParam result
        uint8_t dxl_error = 0;              // Dynamixel error
        

        // Send to pupil. No feedback

        index = PUPIL_INDEX_IO;

        for (int i = PUPIL_ID_MIN; i <= PUPIL_ID_MAX; i++)
        {
            /*if (!torqueEnable.empty())
            {
                uint8_t param_default = torqueEnable[index];
                if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, i, 24, param_default, &dxl_error))
                {
                    // Notify(msg_warning, "[ID:%03d] write1ByteTxRx failed", i);
                    portHandlerPupil->clearPort();
                    return false;
                }
                */

            if (goalPosition.connected())
            {
                uint16_t param_default = goalPosition[index]; // Not using degrees.
                // Goal postiion feature/bug. If torque enable = 0 and goal position is sent. Torque enable will be 1.
                if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, i, 30, param_default, &dxl_error))
                {
                    Notify(msg_warning, std::string("[ID:" + std::to_string(i) + "] write2ByteTxRx failed"));
                    portHandlerPupil->clearPort();
                    return false;
                }
            }
            else
            {
                Notify(msg_fatal_error, "Running module without a goal position input is not supported.");
                return false;
            }
            // XL 320 has no current position mode. Ignores goal current input
            // No feedback from pupils. Also no temperature check. Bad idea?
            index++;
        }
        return (true);
    }

    bool Communication(int IDMin, int IDMax, int IOIndex, dynamixel::PortHandler *portHandler, dynamixel::PacketHandler *packetHandler, dynamixel::GroupSyncRead *groupSyncRead, dynamixel::GroupSyncWrite *groupSyncWrite)
    {
        if (portHandler == NULL) // If no port handler return true. Only return false if communication went wrong.
            return true;

        int index = 0;
        int dxl_comm_result = COMM_TX_FAIL; // Communication result
        bool dxl_addparam_result = false;   // addParam result
        bool dxl_getdata_result = false;    // GetParam result

        uint8_t dxl_error = 0;       // Dynamixel error
        uint8_t param_sync_write[len_write_data]; // 7 byte sync write is not supported for the DynamixelSDK

        int32_t dxl_present_position = 0;
        int16_t dxl_present_current = 0;
        int8_t dxl_present_temperature = 0;
        uint16_t dxl_goal_current = 0;

        // Add id for syncread
        for (int i = IDMin; i <= IDMax; i++)
        {
            if (!groupSyncRead->addParam(i))
            {
                groupSyncWrite->clearParam();
                groupSyncRead->clearParam();
                return false;
            }
        }

        // Sync read
        dxl_comm_result = groupSyncRead->txRxPacket();
        if (dxl_comm_result != COMM_SUCCESS)
        {
            Warning("GroupSyncRead failed: ");
            groupSyncWrite->clearParam();
            groupSyncRead->clearParam();
            return false;
        }

        // Check if data is available
        for (int i = IDMin; i <= IDMax; i++)
        {
            // Corrected isAvailable check: Use INDIRECTDATA_FOR_READ (634) and correct length 7
            dxl_comm_result = groupSyncRead->isAvailable(i, INDIRECTDATA_FOR_READ, 4 + 2 + 1);
            if (!dxl_comm_result)
            {
                Notify(msg_warning, "SyncRead data not available for ID: " + std::to_string(i) + " at indirect data addr " + std::to_string(INDIRECTDATA_FOR_READ));
                groupSyncWrite->clearParam();
                groupSyncRead->clearParam();
                return false;
            }
        }

        // Extract data
        index = IOIndex;
        for (int i = IDMin; i <= IDMax; i++)
        {
            // Read 4 bytes starting at INDIRECTDATA_FOR_READ (634)
            dxl_present_position = groupSyncRead->getData(i, INDIRECTDATA_FOR_READ, 4);    // Present position
            // Read 2 bytes starting at INDIRECTDATA_FOR_READ + 4 = 638
            dxl_present_current = groupSyncRead->getData(i, INDIRECTDATA_FOR_READ + 4, 2); // Present current (Offset 4 from start)
            // TODO: Read temperature at offset 6 (INDIRECTDATA_FOR_READ + 6 = 640)
            
            presentPosition[index] = dxl_present_position / 4096.0 * 360.0; // degrees
            // Use correct conversion factor for Present Current
            presentCurrent[index] = dxl_present_current * CURRENT_UNIT; // mA
            index++;
        }

        // Send (sync write)
        index = IOIndex;
        for (int i = IDMin; i <= IDMax; i++)
        {
            param_sync_write[0] = torqueEnable(i); // Torque on

            if (goalPosition.connected())
            {
                int value = goalPosition[index] / 360.0 * 4096.0;
                param_sync_write[1] = DXL_LOBYTE(DXL_LOWORD(value));
                param_sync_write[2] = DXL_HIBYTE(DXL_LOWORD(value));
                param_sync_write[3] = DXL_LOBYTE(DXL_HIWORD(value));
                param_sync_write[4] = DXL_HIBYTE(DXL_HIWORD(value));
            }
            else
            {
                Notify(msg_fatal_error, "Running module without a goal position input is not supported.");
                groupSyncWrite->clearParam();
                groupSyncRead->clearParam();
                return false;
            }
            if (goalCurrent.connected() && controlMode == "CurrentPosition")
            {
                // Use the correct conversion factor for Goal Current
                int value_current = goalCurrent[index] / CURRENT_UNIT; 
                param_sync_write[5] = DXL_LOBYTE(value_current);
                param_sync_write[6] = DXL_HIBYTE(value_current);
            }
            else // THIS IS TRUE FOR THE NON-WORKING CASE
            {
                // Send 0 for Goal Current bytes, as it's likely unused/problematic in pure Position Mode
                param_sync_write[5] = 0; 
                param_sync_write[6] = 0;
            }
            if (goalPWM.connected() && controlMode == "Position")
            {
                int value_pwm = goalPWM[index] / 0.11299;
                param_sync_write[7] = DXL_LOBYTE(value_pwm);
                param_sync_write[8] = DXL_HIBYTE(value_pwm);
                Debug("Goal PWM: " + std::to_string(goalPWM[index]));
            }
            else
            {
                int value_pwm = 100 / 0.11299; // 100% PWM
                param_sync_write[7] = DXL_LOBYTE(value_pwm);
                param_sync_write[8] = DXL_HIBYTE(value_pwm);
            }

            dxl_addparam_result = groupSyncWrite->addParam(i, param_sync_write);
            if (!dxl_addparam_result)
            {
                std::cout << "addParam failed" << std::endl;

                groupSyncWrite->clearParam();
                groupSyncRead->clearParam();
                return false;
            }

            index++;
        }

        // Syncwrite
        dxl_comm_result = groupSyncWrite->txPacket();
        if (dxl_comm_result != COMM_SUCCESS)
        {
            groupSyncWrite->clearParam();
            groupSyncRead->clearParam();
            std::cout << "Sync failed" << std::endl;
            return false;
        }

        groupSyncWrite->clearParam();
        groupSyncRead->clearParam();
        return true;
    }

   
    bool ParameterJsonFileExists(std::string robotType, std::string controlMode){
        // Construct the filename
        std::string filename = "ServoParameters" + robotType + "_" + controlMode + ".json";
        std::string path = __FILE__;
        // Remove the filename from the path
        path = path.substr(0, path.find_last_of("/\\"));
        filename = path + "/" + filename;
        return std::filesystem::exists(filename);
    }
    
    matrix ReadJsonToMatrix(int minID, int maxID, std::string robotType, std::string servoChain, std::string controlMode){       
        int numParameters = 0;
        std::string tunedParameters;
        // Construct the filename
        std::string filename = "ServoParameters" + robotType + "_" + controlMode + ".json";
        std::string path = __FILE__;
        // Remove the filename from the path
        path = path.substr(0, path.find_last_of("/\\"));
        filename = path + "/" + filename;
        
        // Initialize JSON object
        nlohmann::json jsonData;

        std::ifstream infile(filename);

        // Read existing JSON data if the file exists
        if (infile.is_open() && infile.peek() != std::ifstream::traits_type::eof()) {
            infile >> jsonData;
            infile.close();
        }
        else{
            Error("Parameter file " + filename + " does not exist.");
            return matrix();
        }
       
  
        nlohmann::json& robotData = jsonData[robotType];
        nlohmann::json& servoChainData = robotData[servoChain];

        
        // Determine the number of parameters
        for (auto& chain : servoChainData) {
            if (chain.contains("servoID")) {
                numParameters = chain.size() -1; // exclude servoID
                break;
            }
        }

        // Initialize the result matrix
        matrix result(maxID - minID + 1, numParameters); // all servo chains start from ID 2

        // Check if the servo ID 
        for (auto& chain : servoChainData) {
            int i = 0;
            for (int ID = minID; ID <= maxID; ID++){
                if (chain["servoID"] == ID) {
                    printf("ID: %d\n", ID);
                    //loop through all keys of the chain
                    int paramIndex = 0;
                    for (auto& it : chain.items()) {
                        if(it.key() != "servoID"){
                            //only add to tuned parameter the one time
                            if (ID == minID) {
                            parameter_lst.push_back(it.key());
                            tunedParameters += "\"" + it.key() + "\", ";
                            } 
                        result(ID - minID, paramIndex) = it.value();
                        }
                        paramIndex++;
                    }
                }
                i++; 
            }           
        }
        result.set_labels(0, tunedParameters);
        return result;
    }

    void CreateParameterMatrices(){

        headData = ReadJsonToMatrix(HEAD_ID_MIN, HEAD_ID_MAX, robot[robotName].type, "Head", controlMode);
       

        if (EpiFullMode){
            bodyData = ReadJsonToMatrix(BODY_ID_MIN, BODY_ID_MAX, robot[robotName].type, "Body", controlMode);
            leftArmData = ReadJsonToMatrix(ARM_ID_MIN, ARM_ID_MAX, robot[robotName].type, "LeftArm", controlMode);
            rightArmData = ReadJsonToMatrix(ARM_ID_MIN, ARM_ID_MAX, robot[robotName].type, "RightArm", controlMode);
        }
        
    }


    void Init()
    {

        // Robots configurations
        robot["EpiPink"] = {.serialPortPupil = "/dev/cu.usbserial-FT66U0T9",
                            .serialPortHead = "/dev/cu.usbserial-FT66WMQF",
                            .serialPortBody = "",
                            .serialPortLeftArm = "",
                            .serialPortRightArm = "",
                            .type = "EpiTorso"};

        robot["EpiWhite"] = {.serialPortPupil = "/dev/cu.usbserial-FT66WV4A",
                             .serialPortHead = "/dev/cu.usbserial-FT6S4JL9",
                             .serialPortBody = "",
                             .serialPortLeftArm = "",
                             .serialPortRightArm = "",
                             .type = "EpiTorso"};

        robot["EpiRed"] = {.serialPortPupil = "/dev/cu.usbserial-FT4TCJXI",
                           .serialPortHead = "/dev/cu.usbserial-FT4TCGUT",
                           .serialPortBody = "",
                           .serialPortLeftArm = "",
                           .serialPortRightArm = "",
                           .type = "EpiTorso"};

        robot["EpiRedDemo"] = {.serialPortPupil = "/dev/ttyUSB0",
                               .serialPortHead = "/dev/ttyUSB1",
                               .serialPortBody = "",
                               .serialPortLeftArm = "",
                               .serialPortRightArm = "",
                               .type = "EpiTorso"};

        robot["EpiOrange"] = {.serialPortPupil = "/dev/cu.usbserial-FT3WI2WH",
                              .serialPortHead = "/dev/cu.usbserial-FT3WI2K2",
                              .serialPortBody = "",
                              .serialPortLeftArm = "",
                              .serialPortRightArm = "",
                              .type = "EpiTorso"};

        robot["EpiYellow"] = {.serialPortPupil = "/dev/cu.usbserial-FT6S4IFI",
                              .serialPortHead = "/dev/cu.usbserial-FT6RW7PH",
                              .serialPortBody = "",
                              .serialPortLeftArm = "",
                              .serialPortRightArm = "",
                              .type = "EpiTorso"};

        robot["EpiGreen"] = {.serialPortPupil = "/dev/cu.usbserial-FT6S4JMH",
                             .serialPortHead = "/dev/cu.usbserial-FT66WT6W",
                             .serialPortBody = "",
                             .serialPortLeftArm = "",
                             .serialPortRightArm = "",
                             .type = "EpiTorso"};

        robot["EpiBlue"] = {.serialPortPupil = "/dev/cu.usbserial-FT66WS1F",
                            .serialPortHead = "/dev/cu.usbserial-FT4THUNJ",
                            .serialPortBody = "/dev/cu.usbserial-FT4THV1M",
                            .serialPortLeftArm = "/dev/cu.usbserial-FT4TFSV0",
                            .serialPortRightArm = "/dev/cu.usbserial-FT4TCVTT",
                            .type = "Epi"};

        robot["EpiGray"] = {.serialPortPupil = "/dev/cu.usbserial-FT6S4JL5",
                            .serialPortHead = "/dev/cu.usbserial-FT66WV43",
                            .serialPortBody = "",
                            .serialPortLeftArm = "",
                            .serialPortRightArm = "",
                            .type = "EpiTorso"};

        robot["EpiBlack"] = {.serialPortPupil = "/dev/cu.usbserial-FT66WIVC",
                             .serialPortHead = "/dev/cu.usbserial-FT3WHSCR",
                             .serialPortBody = "",
                             .serialPortLeftArm = "",
                             .serialPortRightArm = "",
                             .type = "EpiTorso"};

        robotName = GetValue("robot");

        // Check if robotname exist in configuration
        if (robot.find(robotName) == robot.end())
        {
            Error(std::string("%s is not supported") + robotName);
            return;
        }

        // Check type of robot
        EpiTorsoMode = (robot[robotName].type.compare("EpiTorso") == 0);
        EpiFullMode = (robot[robotName].type.compare("Epi") == 0);

        Notify(msg_debug, std::string("Connecting to " + robotName + " (" + robot[robotName].type + ")"));

        std::string sTable = R"({"Torque Enable": {"Address": 64,"Bytes": 1},
                                    "LED": {"Address": 65,"Bytes": 1},
                                    "Status Return Level": {
                                        "Address": 68,
                                        "Bytes": 1
                                    },
                                    "Registered Instruction": {
                                        "Address": 69,
                                        "Bytes": 1
                                    },
                                    "Hardware Error Status": {
                                        "Address": 70,
                                        "Bytes": 1
                                    },
                                    "Velocity I Gain": {
                                        "Address": 76,
                                        "Bytes": 2
                                    },
                                    "Velocity P Gain": {
                                        "Address": 78,
                                        "Bytes": 2
                                    },
                                    "Position D Gain": {
                                        "Address": 80,
                                        "Bytes": 2
                                    },
                                    "Position I Gain": {
                                        "Address": 82,
                                        "Bytes": 2
                                    },
                                    "Position P Gain": {
                                        "Address": 84,
                                        "Bytes": 2
                                    },
                                    "Feedforward 2nd Gain": {
                                        "Address": 88,
                                        "Bytes": 2
                                    },
                                    "Feedforward 1st Gain": {
                                        "Address": 90,
                                        "Bytes": 2
                                    },
                                    "BUS Watchdog": {
                                        "Address": 98,
                                        "Bytes": 1
                                    },
                                    "Goal PWM": {
                                        "Address": 100,
                                        "Bytes": 2
                                    },
                                    "Goal Current": {
                                        "Address": 102,
                                        "Bytes": 2
                                    },
                                    "Goal Velocity": {
                                        "Address": 104,
                                        "Bytes": 4
                                    },
                                    "Profile Acceleration": {
                                        "Address": 108,
                                        "Bytes": 4
                                    },
                                    "Profile Velocity": {
                                        "Address": 112,
                                        "Bytes": 4
                                    },
                                    "Goal Position": {
                                        "Address": 116,
                                        "Bytes": 4
                                    },
                                    "Realtime Tick": {
                                        "Address": 120,
                                        "Bytes": 2
                                    },
                                    "Moving": {
                                        "Address": 122,
                                        "Bytes": 1
                                    },
                                    "Moving Status": {
                                        "Address": 123,
                                        "Bytes": 1
                                    },
                                    "Present PWM": {
                                        "Address": 124,
                                        "Bytes": 2
                                    },
                                    "Present Current": {
                                        "Address": 126,
                                        "Bytes": 2
                                    },
                                    "Present Velocity": {
                                        "Address": 128,
                                        "Bytes": 4
                                    },
                                    "Present Position": {
                                        "Address": 132,
                                        "Bytes": 4
                                    },
                                    "Velocity Trajectory": {
                                        "Address": 136,
                                        "Bytes": 4
                                    },
                                    "Position Trajectory": {
                                        "Address": 140,
                                        "Bytes": 4
                                    },
                                    "Present Input Voltage": {
                                        "Address": 144,
                                        "Bytes": 2
                                    },
                                    "Present Temperature": {
                                        "Address": 146,
                                        "Bytes": 1
                                    }
                                    }
                                    )" ;
        
        servoControlTable = parse_json(sTable);  
        //Ikaros parameters
        Bind(minLimitPosition, "MinLimitPosition");
        Bind(maxLimitPosition, "MaxLimitPosition");
        Bind(ServoControlMode, "ServoControlMode");
        controlMode = ServoControlMode.as_string();
        Bind(dataToWrite, "DataToWrite");
        
        // Ikaros input
        Bind(goalPosition, "GOAL_POSITION");
        Bind(goalCurrent, "GOAL_CURRENT");
        Bind(goalPWM, "GOAL_PWM");
        Bind(torqueEnable, "TORQUE_ENABLE");

        // Ikaros output
        Bind(presentPosition, "PRESENT_POSITION");
        Bind(presentCurrent, "PRESENT_CURRENT");

        std::cout << "EpiServos: " << robotName << std::endl;

        len_write_data = CalculateLenWriteData(dataToWrite.as_string());
        Debug("Bytes to write: " + std::to_string(len_write_data));
        Debug("Data to write: " + dataToWrite.as_string());

       
        // Check if the input size are correct. We do not need to have an input at all!
        if (EpiTorsoMode)
        {
            if (goalPosition.connected())
                if (goalPosition.size() < EPI_TORSO_NR_SERVOS)
                    Notify(msg_fatal_error, "Input size goal position does not match robot type\n");
            if (goalCurrent.connected())
                if (goalCurrent.size() < EPI_TORSO_NR_SERVOS)
                    Notify(msg_fatal_error, "Input size goal current does not match robot type\n");
            /* if (!torqueEnable.empty())
                 if (torqueEnable.size() < EPI_TORSO_NR_SERVOS)
                     Notify(msg_fatal_error, "Input size torque enable does not match robot type\n");*/
        }
        else if (EpiFullMode)
        {
            if (goalPosition.connected())
                if (goalPosition.size() < EPI_NR_SERVOS)
                    Notify(msg_fatal_error, "Input size goal position does not match robot type\n");
            if (goalCurrent.connected())
                if (goalCurrent.size() < EPI_NR_SERVOS)
                    Notify(msg_fatal_error, "Input size goal current does not match robot type\n");
            /*if (!torqueEnable.empty())
                if (torqueEnable.size() < EPI_NR_SERVOS)
                    Notify(msg_fatal_error, "Input size torque enable does not match robot type\n");*/
        }

        // Ikaros parameter simulate
        Bind(simulate, "simulate");

        if (simulate)
        {
            Notify(msg_print, "Simulate servos");
            return;
        }

        // Epi torso
        // =========
        if (EpiTorsoMode || EpiFullMode)
        {
            int dxl_comm_result;
            std::vector<uint8_t> vec;

            // Neck/Eyes (id 2,3) =  2x MX106R Eyes = 2xMX28R (id 3,4)

            // Init Dynaxmixel SDK
            portHandlerHead = dynamixel::PortHandler::getPortHandler(robot[robotName].serialPortHead.c_str());
            packetHandlerHead = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

            Notify(msg_debug, "Setting up serial port (head)");

            portHandlers = {portHandlerHead};
            packetHandlers = {packetHandlerHead};

            // Open port
            if (portHandlerHead->openPort())
                Notify(msg_debug, "Succeeded to open serial port!");
            else
            {
                Notify(msg_fatal_error, "Failed to open serial port!");
                return;
            }

            // Set port baudrate
            if (portHandlerHead->setBaudRate(BAUDRATE3M))
                Notify(msg_debug, "Succeeded to change baudrate!");
            else
            {
                Notify(msg_fatal_error, "Failed to change baudrate!");
                return;
            }

            // Ping all the servos to make sure they are all there.
            dxl_comm_result = packetHandlerHead->broadcastPing(portHandlerHead, vec);
            if (dxl_comm_result != COMM_SUCCESS)
                Notify(msg_warning, "Can not send out broadcast ping to servos");

            Notify(msg_debug, "Detected Dynamixel (Head): ");
            for (int i = 0; i < (int)vec.size(); i++)
                Notify(msg_debug, std::string("[ID: " + std::to_string(vec.at(i)) + "]"));

            // Pupil (id 2,3) = XL320 Left eye, right eye

            // Init Dynaxmixel SDK
            portHandlerPupil = dynamixel::PortHandler::getPortHandler(robot[robotName].serialPortPupil.c_str());
            packetHandlerPupil = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

            Notify(msg_debug, "Setting up serial port (Pupil)");

            // Open port
            if (portHandlerPupil->openPort())
                Notify(msg_debug, "Succeeded to open serial port!");
            else
            {
                Notify(msg_fatal_error, "Failed to open serial port!");
                return;
            }
            // Set port baudrate
            if (portHandlerPupil->setBaudRate(BAUDRATE1M))
                Notify(msg_debug, "Succeeded to change baudrate!");
            else
            {
                Notify(msg_fatal_error, "Failed to change baudrate!");
                return;
            }
            // Ping all the servos to make sure they are all there.
            dxl_comm_result = packetHandlerPupil->broadcastPing(portHandlerPupil, vec);
            if (dxl_comm_result != COMM_SUCCESS)
                Notify(msg_warning, "Can not send out broadcast ping to servos");

            Notify(msg_debug, "Detected Dynamixel (Pupil): ");
            for (int i = 0; i < (int)vec.size(); i++)
                Notify(msg_debug, std::string("[ID: " + std::to_string(vec.at(i)) + "]"));
        }
        else
        {
            Notify(msg_fatal_error, "Robot type is not yet implementet");
        }
        if (EpiFullMode)
        {
            int dxl_comm_result;
            std::vector<uint8_t> vec;

            // Left arm 6x MX106R 1 MX28R

            // Init Dynaxmixel SDK
            portHandlerLeftArm = dynamixel::PortHandler::getPortHandler(robot[robotName].serialPortLeftArm.c_str());
            packetHandlerLeftArm = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

            Notify(msg_debug, "Setting up serial port (Left arm)");

            // Open port
            if (portHandlerLeftArm->openPort())
                Notify(msg_debug, "Succeeded to open serial port!");
            else
            {
                Notify(msg_fatal_error, "Failed to open serial port!");
                return;
            }
            // Set port baudrate
            if (portHandlerLeftArm->setBaudRate(BAUDRATE3M))
                Notify(msg_debug, "Succeeded to change baudrate!");
            else
            {
                Notify(msg_fatal_error, "Failed to change baudrate!");
                return;
            }
            // Ping all the servos to make sure they are all there.
            dxl_comm_result = packetHandlerLeftArm->broadcastPing(portHandlerLeftArm, vec);
            if (dxl_comm_result != COMM_SUCCESS)
                Notify(msg_warning, "Can not send out broadcast ping to servos");

            Notify(msg_debug, "Detected Dynamixel (Left arm): ");
            for (int i = 0; i < (int)vec.size(); i++)
                Notify(msg_debug, std::string("[ID: " + std::to_string(vec.at(i)) + "]"));

            // Right arm 6x MX106R 1 MX28R

            // Init Dynaxmixel SDK
            portHandlerRightArm = dynamixel::PortHandler::getPortHandler(robot[robotName].serialPortRightArm.c_str());
            packetHandlerRightArm = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

            Notify(msg_debug, "Setting up serial port (Right arm)");

            // Open port
            if (portHandlerRightArm->openPort())
                Notify(msg_debug, "Succeeded to open serial port!");
            else
            {
                Notify(msg_fatal_error, "Failed to open serial port!");
                return;
            }
            // Set port baudrate
            if (portHandlerRightArm->setBaudRate(BAUDRATE3M))
                Notify(msg_debug, "Succeeded to change baudrate!");
            else
            {
                Notify(msg_fatal_error, "Failed to change baudrate!");
                return;
            }
            // Ping all the servos to make sure they are all there.
            dxl_comm_result = packetHandlerRightArm->broadcastPing(portHandlerRightArm, vec);
            if (dxl_comm_result != COMM_SUCCESS)
                Notify(msg_warning, "Can not send out broadcast ping to servos");

            Notify(msg_debug, "Detected Dynamixel (Right arm): ");
            for (int i = 0; i < (int)vec.size(); i++)
                Notify(msg_debug, std::string("[ID: " + std::to_string(vec.at(i)) + "]"));

            // Body MX106R

            // Init Dynaxmixel SDK
            portHandlerBody = dynamixel::PortHandler::getPortHandler(robot[robotName].serialPortBody.c_str());
            packetHandlerBody = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

            Notify(msg_debug, "Setting up serial port (Body)");

            // Open port
            if (portHandlerBody->openPort())
                Notify(msg_debug, "Succeeded to open serial port!");
            else
            {
                Notify(msg_fatal_error, "Failed to open serial port!");
                return;
            }
            // Set port baudrate
            if (portHandlerBody->setBaudRate(BAUDRATE3M))
                Notify(msg_debug, "Succeeded to change baudrate!");
            else
            {
                Notify(msg_fatal_error, "Failed to change baudrate!");
                return;
            }
            // Ping all the servos to make sure they are all there.
            dxl_comm_result = packetHandlerBody->broadcastPing(portHandlerBody, vec);
            if (dxl_comm_result != COMM_SUCCESS)
                Notify(msg_warning, "Can not send out broadcast ping to servos");

            Notify(msg_debug, "Detected Dynamixel (Body): \n");
  
            portHandlers = {portHandlerHead, portHandlerLeftArm, portHandlerRightArm, portHandlerBody};
            packetHandlers = {packetHandlerHead, packetHandlerLeftArm, packetHandlerRightArm, packetHandlerBody};
        }

        // Create dynamixel objects
        if (EpiTorsoMode || EpiFullMode)
        {
            groupSyncWriteHead = new dynamixel::GroupSyncWrite(portHandlerHead, packetHandlerHead, INDIRECTDATA_FOR_WRITE, len_write_data);   // Start=224, Len=7 (Torque(1)+Pos(4)+Current(2))
            // Corrected GroupSyncRead initialization: Start=634, Len=7 (Pos(4)+Current(2)+Temp(1))
            groupSyncReadHead = new dynamixel::GroupSyncRead(portHandlerHead, packetHandlerHead, INDIRECTDATA_FOR_READ, 4 + 2 + 1); 
        }
        if (EpiFullMode)
        {
            // Apply similar corrections for Arms and Body
            groupSyncWriteLeftArm = new dynamixel::GroupSyncWrite(portHandlerLeftArm, packetHandlerLeftArm, INDIRECTDATA_FOR_WRITE, len_write_data);
            groupSyncReadLeftArm = new dynamixel::GroupSyncRead(portHandlerLeftArm, packetHandlerLeftArm, INDIRECTDATA_FOR_READ, 4 + 2 + 1);
            groupSyncWriteRightArm = new dynamixel::GroupSyncWrite(portHandlerRightArm, packetHandlerRightArm, INDIRECTDATA_FOR_WRITE, len_write_data);
            groupSyncReadRightArm = new dynamixel::GroupSyncRead(portHandlerRightArm, packetHandlerRightArm, INDIRECTDATA_FOR_READ, 4 + 2 + 1);
            groupSyncWriteBody = new dynamixel::GroupSyncWrite(portHandlerBody, packetHandlerBody, INDIRECTDATA_FOR_WRITE, len_write_data);
            groupSyncReadBody = new dynamixel::GroupSyncRead(portHandlerBody, packetHandlerBody, INDIRECTDATA_FOR_READ, 4 + 2 + 1);
            // groupSyncWritePupil = new dynamixel::GroupSyncWrite(portHandlerPupil, packetHandlerPupil, 30, 2); 
        }

        if(ParameterJsonFileExists(robot[robotName].type, controlMode)){
            std::cout << "Reading json parameter file" << std::endl;
            
            CreateParameterMatrices();
            Notify(msg_trace, "Setting servo settings"); 
            SetServoSettings(); // TODO: Check return value? Assumed it prints errors and returns false if needed.
            Notify(msg_trace, "Setting min max limits");
            // Check return value of SetMinMaxLimits
            if (!SetMinMaxLimits()) {
                Notify(msg_fatal_error, "Failed to set min/max hardware limits on servos.");
                return; // Stop initialization if limits failed
            }

        }
        else{
            Notify(msg_warning, "No parameter file found for this robot type. Using default settings.");
            if (!SetDefaultSettingServo())
                Notify(msg_fatal_error, "Unable to write default settings on servos\n");
        }

        AutoCalibratePupil();

        Notify(msg_debug, "torque down servos and prepering servos for write defualt settings\n");
        if (!PowerOffRobot())
            Notify(msg_fatal_error, "Unable torque down servos\n");
        if (!PowerOnRobot())
            Notify(msg_fatal_error, "Unable torque up servos");
    }

    float PupilMMToDynamixel(float mm, int min, int max)
    {
        // Quick fix
        float minMM = 4.2;
        float maxMM = 19.7;
        float deltaMM = maxMM - minMM;

        float minDeg = min;
        float maxDeg = max;
        float deltDeg = maxDeg - minDeg;

        if (mm < minMM)
            mm = minMM;
        if (mm > maxMM)
            mm = maxMM;

        return (-(mm - minMM) / deltaMM * deltDeg + maxDeg); // Higher goal position gives smaller pupil
    }

    void Tick()
    {   
       
        goalPosition[PUPIL_INDEX_IO] = clip(goalPosition[PUPIL_INDEX_IO], 5, 16); // Pupil size must be between 5 mm to 16 mm.
        goalPosition[PUPIL_INDEX_IO + 1] = clip(goalPosition[PUPIL_INDEX_IO + 1], 5, 16); // Pupil size must be between 5 mm to 16 mm.

        // Special case. As pupil does not have any feedback we just return goal position
        presentPosition[PUPIL_INDEX_IO]    =     goalPosition[PUPIL_INDEX_IO];
        presentPosition[PUPIL_INDEX_IO+1]  =     goalPosition[PUPIL_INDEX_IO+1];

       
        if (goalPosition.connected() && goalPosition[0] > 0){
            for (int i = 0; i < EPI_TORSO_NR_SERVOS -2; i++) // -2 for the pupils
            {
                goalPosition[i] = clip(goalPosition[i], minLimitPosition[i], maxLimitPosition[i]);
                Debug("Goal position: " + std::to_string(goalPosition[i]));
            }
        }
        if (simulate)
        {

            //reset_array(presentCurrent, presentCurrentSize); // 0 mA

            float maxVel = 45.0 / 1000 * 10; // No timebase

            // Comment out GetTickLength() Adding constant.
            // if (GetTickLength() != 0)
            //    maxVel = 45.0 / 1000 * GetTickLength(); // Maximum change in one second in degrees / timebase
            maxVel = 0.01; // This should be a ratio parameter

            for (int i = 0; i < EPI_NR_SERVOS; i++)
            {
                if (goalPosition.connected())
                    if (std::isnan(goalPosition[i]))
                    {
                        Notify(msg_warning, "EpiServo module position input has NAN\n");
                        return;
                    }
                if (goalCurrent.connected())
                    if (std::isnan(goalCurrent[i]))
                    {
                        Notify(msg_warning, "EpiServo module current input has NAN\n");
                        return;
                    }

                if (goalPosition.connected())
                    presentPosition[i] = presentPosition[i] + 0.9 * (clip(goalPosition(i) - presentPosition(i), -maxVel, maxVel)); // adding some smoothing to prevent oscillation in simulation mode

                if (goalCurrent.connected())
                    presentCurrent(i) = presentCurrent(i) + 0.06 * (goalCurrent(i) - presentCurrent(i));
            }
            return;
        }


        // dictionary d; // is this even used. Did not exist with I started this module. Perhaps I should make use of the dictionary now.

        // Special case for pupil uses mm instead of degrees. Also clip the angles if it outside the calibrated range.
        goalPosition[PUPIL_INDEX_IO] = PupilMMToDynamixel(goalPosition[PUPIL_INDEX_IO], AngleMinLimitPupil[0], AngleMaxLimitPupil[0]);
        goalPosition[PUPIL_INDEX_IO + 1] = PupilMMToDynamixel(goalPosition[PUPIL_INDEX_IO + 1], AngleMinLimitPupil[1], AngleMaxLimitPupil[1]);

        
            // Fire up some threads to work in parallell
        auto headThread = std::async(std::launch::async, &EpiServos::Communication, this, HEAD_ID_MIN, HEAD_ID_MAX, HEAD_INDEX_IO, std::ref(portHandlerHead), std::ref(packetHandlerHead), std::ref(groupSyncReadHead), std::ref(groupSyncWriteHead));
        auto pupilThread = std::async(std::launch::async, &EpiServos::CommunicationPupil, this); // Special!

        if (!headThread.get())
        {
            Warning("Can not communicate with head");
            portHandlerHead->clearPort();
        }

        if (!pupilThread.get())
        {
            Warning("Oops.. Communication glitch with pupil servo");
            portHandlerPupil->clearPort();
        }

        if (EpiFullMode)
        {
            auto leftArmThread = std::async(std::launch::async, &EpiServos::Communication, this, ARM_ID_MIN, ARM_ID_MAX, LEFT_ARM_INDEX_IO, std::ref(portHandlerLeftArm), std::ref(packetHandlerLeftArm), std::ref(groupSyncReadLeftArm), std::ref(groupSyncWriteLeftArm));
            auto rightArmThread = std::async(std::launch::async, &EpiServos::Communication, this, ARM_ID_MIN, ARM_ID_MAX, RIGHT_ARM_INDEX_IO, std::ref(portHandlerRightArm), std::ref(packetHandlerRightArm), std::ref(groupSyncReadRightArm), std::ref(groupSyncWriteRightArm));
            auto bodyThread = std::async(std::launch::async, &EpiServos::Communication, this, BODY_ID_MIN, BODY_ID_MIN, BODY_INDEX_IO, std::ref(portHandlerBody), std::ref(packetHandlerBody), std::ref(groupSyncReadBody), std::ref(groupSyncWriteBody));

            if (!leftArmThread.get())
            {
                Warning("Can not communicate with left arm");
                portHandlerLeftArm->clearPort();
            }
            if (!rightArmThread.get())
            {
                Warning("Can not communicate with right arm");
                portHandlerRightArm->clearPort();
            }
            if (!bodyThread.get())
            {
                Warning("Can not communicate with body");
                portHandlerBody->clearPort();
            }
        }
    }

    // A function that set importat parameters in the control table.
    // Baud rate and ID needs to be set manually.
    bool SetDefaultSettingServo() {
        
        uint32_t param_default_4Byte;
        uint32_t profile_acceleration = 50;
        uint32_t profile_velocity = 210;

        uint16_t p_gain_head = 850;
        uint16_t i_gain_head = 0;
        uint16_t d_gain_head = 0;

        uint16_t p_gain_arm = 850;
        uint16_t i_gain_arm = 0;
        uint16_t d_gain_arm = 0;

        uint16_t p_gain_body = 850;
        uint16_t i_gain_body = 0;
        uint16_t d_gain_body = 0;

        uint16_t pupil_moving_speed = 150;
        uint8_t param_default_1Byte;
        uint8_t pupil_p_gain = 100;
        uint8_t pupil_i_gain = 20;
        uint8_t pupil_d_gain = 5;

        uint8_t dxl_error = 0;
        int dxl_comm_result = COMM_TX_FAIL;

        Notify(msg_debug, "Setting control table on servos\n");

        // Inderect Torque Enable
        for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++)
        {
            if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerHead, i, IND_ADDR_TORQUE_ENABLE, ADDR_TORQUE_ENABLE, &dxl_error))
            {
                std::cout << "Failed to set torque enable for head servo ID: " << i << std::endl;
                return false;
            }
        }
        if (EpiFullMode)
        {
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
            {
                if (COMM_SUCCESS != packetHandlerLeftArm->write2ByteTxRx(portHandlerLeftArm, i, IND_ADDR_TORQUE_ENABLE, ADDR_TORQUE_ENABLE, &dxl_error))
                {
                    std::cout << "Failed to set torque enable for left arm servo ID: " << i << std::endl;
                    return false;
                }
            }
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
            {
                if (COMM_SUCCESS != packetHandlerRightArm->write2ByteTxRx(portHandlerRightArm, i, IND_ADDR_TORQUE_ENABLE, ADDR_TORQUE_ENABLE, &dxl_error))
                {
                    std::cout << "Failed to set torque enable for right arm servo ID: " << i << std::endl;
                    return false;
                }
            }
            for (int i = BODY_ID_MIN; i <= BODY_ID_MAX; i++)
            {
                if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerBody, i, IND_ADDR_TORQUE_ENABLE, ADDR_TORQUE_ENABLE, &dxl_error))
                {
                    std::cout << "Failed to set torque enable for body servo ID: " << i << std::endl;
                    return false;
                }
            }
        }

        // Inderect Goal Position
        for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerHead, i, IND_ADDR_GOAL_POSITION + (2 * j), ADDR_GOAL_POSITION + j, &dxl_error))
                {
                    std::cout << "Failed to set goal position for head servo ID: " << i << ", byte: " << j << std::endl;
                    return false;
                }
            }
        }

        if (EpiFullMode)
        {
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                for (int j = 0; j < 4; j++)
                    if (COMM_SUCCESS != packetHandlerLeftArm->write2ByteTxRx(portHandlerLeftArm, i, IND_ADDR_GOAL_POSITION + (2 * j), ADDR_GOAL_POSITION + j, &dxl_error))
                        return false;
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                for (int j = 0; j < 4; j++)
                    if (COMM_SUCCESS != packetHandlerRightArm->write2ByteTxRx(portHandlerRightArm, i, IND_ADDR_GOAL_POSITION + (2 * j), ADDR_GOAL_POSITION + j, &dxl_error))
                        return false;
            for (int i = BODY_ID_MIN; i <= BODY_ID_MAX; i++)
                for (int j = 0; j < 4; j++)
                    if (COMM_SUCCESS != packetHandlerBody->write2ByteTxRx(portHandlerBody, i, IND_ADDR_GOAL_POSITION + (2 * j), ADDR_GOAL_POSITION + j, &dxl_error))
                        return false;
        }

        // Goal Current
        for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerHead, i, IND_ADDR_GOAL_CURRENT + (2 * j), ADDR_GOAL_CURRENT + j, &dxl_error))
                {
                    std::cout << "Goal current not set for head servo ID: " << i << ", byte: " << j << std::endl;
                    return false;
                }
            }
        }
        // Indirect adress (present position). Feedback
        for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerHead, i, IND_ADDR_PRESENT_POSITION + (2 * j), ADDR_PRESENT_POSITION + j, &dxl_error))
                {
                    std::cout << "Present position not set for head servo ID: " << i << ", byte: " << j << std::endl;
                    return false;
                }
            }
        }
        if (EpiFullMode)
        {
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                for (int j = 0; j < 4; j++)
                    if (COMM_SUCCESS != packetHandlerLeftArm->write2ByteTxRx(portHandlerLeftArm, i, IND_ADDR_PRESENT_POSITION + (2 * j), ADDR_PRESENT_POSITION + j, &dxl_error))
                        return false;
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                for (int j = 0; j < 4; j++)
                    if (COMM_SUCCESS != packetHandlerRightArm->write2ByteTxRx(portHandlerRightArm, i, IND_ADDR_PRESENT_POSITION + (2 * j), ADDR_PRESENT_POSITION + j, &dxl_error))
                        return false;
            for (int i = BODY_ID_MIN; i <= BODY_ID_MAX; i++)
                for (int j = 0; j < 4; j++)
                    if (COMM_SUCCESS != packetHandlerBody->write2ByteTxRx(portHandlerBody, i, IND_ADDR_PRESENT_POSITION + (2 * j), ADDR_PRESENT_POSITION + j, &dxl_error))
                        return false;
        }

        // Indirect adress (present current). Feedback. MX28 does not support current mode. Is this a probelm that we still send this to MX-28
        for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerHead, i, IND_ADDR_PRESENT_CURRENT + (2 * j), ADDR_PRESENT_CURRENT + j, &dxl_error))
                {
                    std::cout << "Present current not set for head servo ID: " << i << ", byte: " << j << std::endl;
                    return false;
                }
            }
        }
        if (EpiFullMode)
        {
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                for (int j = 0; j < 2; j++)
                    if (COMM_SUCCESS != packetHandlerLeftArm->write2ByteTxRx(portHandlerLeftArm, i, IND_ADDR_PRESENT_CURRENT + (2 * j), ADDR_PRESENT_CURRENT + j, &dxl_error))
                        return false;
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                for (int j = 0; j < 2; j++)
                    if (COMM_SUCCESS != packetHandlerRightArm->write2ByteTxRx(portHandlerRightArm, i, IND_ADDR_PRESENT_CURRENT + (2 * j), ADDR_PRESENT_CURRENT + j, &dxl_error))
                        return false;
            for (int i = BODY_ID_MIN; i <= BODY_ID_MAX; i++)
                for (int j = 0; j < 2; j++)
                    if (COMM_SUCCESS != packetHandlerBody->write2ByteTxRx(portHandlerBody, i, IND_ADDR_PRESENT_CURRENT + (2 * j), ADDR_PRESENT_CURRENT + j, &dxl_error))
                        return false;
        }
        // Indirect adress (present temperature).
        for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++)
        {
            if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerHead, i, IND_ADDR_PRESENT_TEMPERATURE, ADDR_PRESENT_TEMPERATURE, &dxl_error))
            {
                std::cout << "Present temperature indir not set for head servo ID: " << i << ", byte: " << std::endl;
                return false;
            }
        }
        if (EpiFullMode)
        {
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerLeftArm->write2ByteTxRx(portHandlerLeftArm, i, IND_ADDR_PRESENT_TEMPERATURE, ADDR_PRESENT_TEMPERATURE, &dxl_error))
                    return false;
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerRightArm->write2ByteTxRx(portHandlerRightArm, i, IND_ADDR_PRESENT_TEMPERATURE, ADDR_PRESENT_TEMPERATURE, &dxl_error))
                    return false;
            for (int i = BODY_ID_MIN; i <= BODY_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerBody->write2ByteTxRx(portHandlerBody, i, IND_ADDR_PRESENT_TEMPERATURE, ADDR_PRESENT_TEMPERATURE, &dxl_error))
                    return false;
        }

        // Profile acceleration

        for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++)
        {
            if (COMM_SUCCESS != packetHandlerHead->write4ByteTxRx(portHandlerHead, i, ADDR_PROFILE_ACCELERATION, profile_acceleration, &dxl_error))
            {
                std::cout << "Profile acceleration for head servo ID: " << i << std::endl;
                return false;
            }
        }
        if (EpiFullMode)
        {
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, i, ADDR_PROFILE_ACCELERATION, profile_acceleration, &dxl_error))
                    return false;
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, i, ADDR_PROFILE_ACCELERATION, profile_acceleration, &dxl_error))
                    return false;
            for (int i = BODY_ID_MIN; i <= BODY_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerBody->write4ByteTxRx(portHandlerBody, i, ADDR_PROFILE_ACCELERATION, profile_acceleration, &dxl_error))
                    return false;
        }

        // Common settings for the servos
        // Profile velocity (210)

        for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++)
        {
            if (COMM_SUCCESS != packetHandlerHead->write4ByteTxRx(portHandlerHead, i, ADDR_PROFILE_VELOCITY, profile_velocity, &dxl_error))
            {
                std::cout << "Profile velocity not set for head servo ID: " << i << std::endl;
                return false;
            }
        }
        if (EpiFullMode)
        {
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, i, ADDR_PROFILE_VELOCITY, profile_velocity, &dxl_error))
                    return false;
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, i, ADDR_PROFILE_VELOCITY, profile_velocity, &dxl_error))
                    return false;
            for (int i = BODY_ID_MIN; i <= BODY_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerBody->write4ByteTxRx(portHandlerBody, i, ADDR_PROFILE_VELOCITY, profile_velocity, &dxl_error))
                    return false;
        }

        // P (850)
        for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++)
        {
            if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerHead, i, ADDR_P, p_gain_head, &dxl_error))
            {
                std::cout << "P (PID) not set for head servo ID: " << i << std::endl;
                return false;
            }
        }
        if (EpiFullMode)
        {
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerLeftArm->write2ByteTxRx(portHandlerLeftArm, i, ADDR_P, p_gain_arm, &dxl_error))
                    return false;
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerRightArm->write2ByteTxRx(portHandlerRightArm, i, ADDR_P, p_gain_arm, &dxl_error))
                    return false;
            for (int i = BODY_ID_MIN; i <= BODY_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerBody->write2ByteTxRx(portHandlerBody, i, ADDR_P, p_gain_body, &dxl_error))
                    return false;
        }

        // I
        // The I value almost killed Epi.
        for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++)
        {
            if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerHead, i, ADDR_I, i_gain_head, &dxl_error))
            {
                std::cout << "I (PID) not set for head servo ID: " << i << std::endl;
                return false;
            }
        }
        if (EpiFullMode)
        {
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerLeftArm->write2ByteTxRx(portHandlerLeftArm, i, ADDR_I, i_gain_arm, &dxl_error))
                    return false;
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerRightArm->write2ByteTxRx(portHandlerRightArm, i, ADDR_I, i_gain_arm, &dxl_error))
                    return false;
            for (int i = BODY_ID_MIN; i <= BODY_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerBody->write2ByteTxRx(portHandlerBody, i, ADDR_I, i_gain_body, &dxl_error))
                    return false;
        }

        // D
        for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++)
        {
            if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerHead, i, ADDR_D, d_gain_head, &dxl_error))
            {
                std::cout << "D (PID) not set for head servo ID: " << i << std::endl;
                return false;
            }
        }
        if (EpiFullMode)
        {
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerLeftArm->write2ByteTxRx(portHandlerLeftArm, i, ADDR_D, d_gain_arm, &dxl_error))
                    return false;
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerRightArm->write2ByteTxRx(portHandlerRightArm, i, ADDR_D, d_gain_arm, &dxl_error))
                    return false;
            for (int i = BODY_ID_MIN; i <= BODY_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerBody->write2ByteTxRx(portHandlerBody, i, ADDR_D, d_gain_body, &dxl_error))
                    return false;
        }

        // Specific setting for each servos
        // HEAD ID 2
        // Limit position max
        uint32_t limit_pos_max_tilt = 2700;
        if (COMM_SUCCESS != packetHandlerHead->write4ByteTxRx(portHandlerHead, 2, 48, limit_pos_max_tilt, &dxl_error))
        {
            std::cout << "Max limit not set for head servo ID: 2 " << std::endl;
            return false;
        }
        // Limit position min
        uint32_t limit_pos_min_tilt = 1300;
        if (COMM_SUCCESS != packetHandlerHead->write4ByteTxRx(portHandlerHead, 2, 52, limit_pos_min_tilt, &dxl_error))
        {
            std::cout << "Min limit not set for head servo ID: 2 " << std::endl;
            return false;
        }

        // HEAD ID 3
        // Limit position max
        uint32_t limit_pos_max_pan = 2500;
        if (COMM_SUCCESS != packetHandlerHead->write4ByteTxRx(portHandlerHead, 3, 48, limit_pos_max_pan, &dxl_error))
            return false;
        // Limit position min
        uint32_t limit_pos_min_pan = 1750;
        if (COMM_SUCCESS != packetHandlerHead->write4ByteTxRx(portHandlerHead, 3, 52, limit_pos_min_pan, &dxl_error))
            return false;

        // HEAD ID 4 (Left eye)
        // Limit position max
        uint32_t limit_pos_max_left_eye = 2300;
        if (COMM_SUCCESS != packetHandlerHead->write4ByteTxRx(portHandlerHead, 4, 48, limit_pos_max_left_eye, &dxl_error))
            return false;
        // Limit position min
        uint32_t limit_pos_min_left_eye = 1830;
        if (COMM_SUCCESS != packetHandlerHead->write4ByteTxRx(portHandlerHead, 4, 52, limit_pos_min_left_eye, &dxl_error))
            return false;

        // HEAD ID 5 (Right eye)
        // Limit position max
        uint32_t limit_pos_max_right_eye = 2200;
        if (COMM_SUCCESS != packetHandlerHead->write4ByteTxRx(portHandlerHead, 5, 48, limit_pos_max_right_eye, &dxl_error))
            return false;

        // Limit position min
        uint32_t limit_pos_min_right_eye = 1780;
        if (COMM_SUCCESS != packetHandlerHead->write4ByteTxRx(portHandlerHead, 5, 52, limit_pos_min_right_eye, &dxl_error))
            return false;

        Timer t;
        double xlTimer = 0.01; // Timer in sec. XL320 need this. Not sure why.

        // PUPIL ID 2 (Left pupil)
        // Limit position min
        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 2, 6, AngleMinLimitPupil[0], &dxl_error))
            return false;
        Sleep(xlTimer);

        // Limit position max
        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 2, 8, AngleMaxLimitPupil[0], &dxl_error))
            return false;
        Sleep(xlTimer);

        // Moving speed
        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 2, 32, pupil_moving_speed, &dxl_error))
            return false;
        Sleep(xlTimer);

        // P
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 2, 29, pupil_p_gain, &dxl_error))
            return false;
        Sleep(xlTimer);

        // I
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 2, 28, pupil_i_gain, &dxl_error))
            return false;
        Sleep(xlTimer);

        // D
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 2, 27, pupil_d_gain, &dxl_error))
            return false;
        Sleep(xlTimer);

        // PUPIL ID 3 (Right pupil)
        // Limit position in
        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 3, 6, AngleMinLimitPupil[1], &dxl_error))
            return false;
        Sleep(xlTimer);

        // Limit position max
        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 3, 8, AngleMaxLimitPupil[1], &dxl_error))
            return false;
        Sleep(xlTimer);

        // Moving speed
        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 3, 32, pupil_moving_speed, &dxl_error))
            return false;
        Sleep(xlTimer);

        // P
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 3, 29, pupil_p_gain, &dxl_error))
            return false;
        Sleep(xlTimer);

        // I
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 3, 28, pupil_i_gain, &dxl_error))
            return false;
        Sleep(xlTimer);

        // D
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 3, 27, pupil_d_gain, &dxl_error))
            return false;
        Sleep(xlTimer);

        if (EpiFullMode)
        {
            // LEFT ARM ID 2
            // Limit position max
            param_default_4Byte = 3200;
            if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, 2, 48, param_default_4Byte, &dxl_error))
                return false;
            // Limit position min
            param_default_4Byte = 600;
            if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, 2, 52, param_default_4Byte, &dxl_error))
                return false;

            // LEFT ARM ID 3
            // Limit position max
            param_default_4Byte = 3200;
            if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, 3, 48, param_default_4Byte, &dxl_error))
                return false;
            // Limit position min
            param_default_4Byte = 800;
            if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, 3, 52, param_default_4Byte, &dxl_error))
                return false;

            // LEFT ARM ID 4
            // Limit position max
            param_default_4Byte = 3000;
            if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, 4, 48, param_default_4Byte, &dxl_error))
                return false;
            // Limit position min
            param_default_4Byte = 1000;
            if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, 4, 52, param_default_4Byte, &dxl_error))
                return false;

            // LEFT ARM ID 5
            // Limit position max
            param_default_4Byte = 2300;
            if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, 5, 48, param_default_4Byte, &dxl_error))
                return false;
            // Limit position min
            param_default_4Byte = 600;
            if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, 5, 52, param_default_4Byte, &dxl_error))
                return false;

            // LEFT ARM ID 6
            // Limit position max
            param_default_4Byte = 3900;
            if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, 6, 48, param_default_4Byte, &dxl_error))
                return false;
            // Limit position min
            param_default_4Byte = 800;
            if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, 6, 52, param_default_4Byte, &dxl_error))
                return false;

            // LEFT ARM ID 7
            // Limit position max
            param_default_4Byte = 4095;
            if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, 7, 48, param_default_4Byte, &dxl_error))
                return false;
            // Limit position min
            param_default_4Byte = 0;
            if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, 7, 52, param_default_4Byte, &dxl_error))
                return false;

            // RIGHT ARM ID 2
            // Limit position max
            param_default_4Byte = 3200;
            if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, 2, 48, param_default_4Byte, &dxl_error))
                return false;
            // Limit position min
            param_default_4Byte = 600;
            if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, 2, 52, param_default_4Byte, &dxl_error))
                return false;
            // RIGHT ARM ID 3
            // Limit position max
            param_default_4Byte = 3300;
            if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, 3, 48, param_default_4Byte, &dxl_error))
                return false;
            // Limit position min
            param_default_4Byte = 900;
            if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, 3, 52, param_default_4Byte, &dxl_error))
                return false;
            // RIGHT ARM ID 4
            // Limit position max
            param_default_4Byte = 3000;
            if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, 4, 48, param_default_4Byte, &dxl_error))
                return false;
            // Limit position min
            param_default_4Byte = 1000;
            if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, 4, 52, param_default_4Byte, &dxl_error))
                return false;
            // RIGHT ARM ID 5
            // Limit position max
            param_default_4Byte = 3600;
            if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, 5, 48, param_default_4Byte, &dxl_error))
                return false;
            // Limit position min
            param_default_4Byte = 1800;
            if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, 5, 52, param_default_4Byte, &dxl_error))
                return false;
            // RIGHT ARM ID 6
            // Limit position max
            param_default_4Byte = 3900;
            if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, 6, 48, param_default_4Byte, &dxl_error))
                return false;
            // Limit position min
            param_default_4Byte = 800;
            if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, 6, 52, param_default_4Byte, &dxl_error))
                return false;
            // RIGHT ARM ID 7
            // Limit position max
            param_default_4Byte = 4095;
            if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, 7, 48, param_default_4Byte, &dxl_error))
                return false;
            // Limit position min
            param_default_4Byte = 0;
            if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, 7, 52, param_default_4Byte, &dxl_error))
                return false;

            // BODY ID 2
            // Limit position max
            param_default_4Byte = 3900;
            if (COMM_SUCCESS != packetHandlerBody->write4ByteTxRx(portHandlerBody, 2, 48, param_default_4Byte, &dxl_error))
                return false;
            // Limit position min
            param_default_4Byte = 100;
            if (COMM_SUCCESS != packetHandlerBody->write4ByteTxRx(portHandlerBody, 2, 52, param_default_4Byte, &dxl_error))
                return false;
        }
        return true; // Yay we manage to set everything we needed.
    }

    bool SetServoSettings() {
        uint32_t param_default_4Byte;
        uint16_t param_default_2Byte;
        uint8_t param_default_1Byte;
        uint8_t dxl_error = 0;
        int default_value = 0;
    
        std::vector<int> maxMinPositionLimitIndex;
        matrix data;
      

        int dxl_comm_result = COMM_TX_FAIL;
        int idMin;
        int idMax;
        int addressRead = INDIRECTADDRESS_FOR_READ;
        int addressWrite = INDIRECTADDRESS_FOR_WRITE;
        int directAddress;

        int byteLength;
        value parameterName;

        Notify(msg_debug, "Setting control table on servos\n");

        //loop through all portHandlers
        for (int p = 0; p < portHandlers.size(); p++) {

            // Ensure p is within the valid range
            if (p < 0 || p >= portHandlers.size() || p >= packetHandlers.size()) {
                std::cout << "Invalid index for portHandlers or packetHandlers: " << p << std::endl;
                return false;
            }

            // Ensure pointers are not null
            if (!portHandlers[p] || !packetHandlers[p]) {
                std::cout << "Null pointer encountered in portHandlers or packetHandlers at index: " << p << std::endl;
                return false;
            }
            //switch statement for different p values. 
            switch (p) {
                case 0:
                idMin = HEAD_ID_MIN;
                idMax = HEAD_ID_MAX;
                data = headData;
                break;
                //same for p==1 and p==2 (right and left arm)
                case 1:
                idMin = ARM_ID_MIN;
                idMax = ARM_ID_MAX;
                data = leftArmData;
                break;
                case 2:
                idMin = ARM_ID_MIN;
                idMax = ARM_ID_MAX;
                data = rightArmData;
                break;
                case 3:
                idMin = BODY_ID_MIN;
                idMax = BODY_ID_MAX;
                data = bodyData;
                break;
            }
         
            for (int id = idMin; id <= idMax; id++) 
            {   
                // Disable Dynamixel Torque :
                // Indirect address would not accessible when the torque is already enabled
                dxl_comm_result = packetHandlers[p]->write1ByteTxRx(portHandlers[p], id, ADDR_TORQUE_ENABLE, 0, &dxl_error);
                if (dxl_comm_result != COMM_SUCCESS)
                {
                    packetHandlers[p]->getTxRxResult(dxl_comm_result);
                }
                else if (dxl_error != 0)
                {
                    packetHandlers[p]->getRxPacketError(dxl_error);
                }
                

                //Setting indirect addresses for all servos
                // Torque Enable
                for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++) {
                    if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerHead, i, IND_ADDR_TORQUE_ENABLE, ADDR_TORQUE_ENABLE, &dxl_error)) {
                        std::cout << "Failed to set torque enable for head servo ID: " << i << std::endl;
                        return false;
                    }
                }
                if (EpiFullMode) {
                    for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++) {
                        if (COMM_SUCCESS != packetHandlerLeftArm->write2ByteTxRx(portHandlerLeftArm, i, IND_ADDR_TORQUE_ENABLE, ADDR_TORQUE_ENABLE, &dxl_error)) {
                            std::cout << "Failed to set torque enable for left arm servo ID: " << i << std::endl;
                            return false;
                        }
                    }
                    for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++) {
                        if (COMM_SUCCESS != packetHandlerRightArm->write2ByteTxRx(portHandlerRightArm, i, IND_ADDR_TORQUE_ENABLE, ADDR_TORQUE_ENABLE, &dxl_error)) {
                            std::cout << "Failed to set torque enable for right arm servo ID: " << i << std::endl;
                            return false;
                        }
                    }
                    for (int i = BODY_ID_MIN; i <= BODY_ID_MAX; i++) {
                        if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerBody, i, IND_ADDR_TORQUE_ENABLE, ADDR_TORQUE_ENABLE, &dxl_error)) {
                            std::cout << "Failed to set torque enable for body servo ID: " << i << std::endl;
                            return false;
                        }
                    }
                } 

                // Goal Position
                for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++) {
                    for (int j = 0; j < 4; j++) {
                        if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerHead, i, IND_ADDR_GOAL_POSITION + (2 * j), ADDR_GOAL_POSITION + j, &dxl_error)) {
                            std::cout << "Failed to set goal position for head servo ID: " << i << ", byte: " << j << std::endl;
                            return false;
                        }
                    }
                }
                if (EpiFullMode)
                {
                    for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                        for (int j = 0; j < 4; j++)
                            if (COMM_SUCCESS != packetHandlerLeftArm->write2ByteTxRx(portHandlerLeftArm, i, IND_ADDR_GOAL_POSITION + (2 * j), ADDR_GOAL_POSITION + j, &dxl_error))
                                return false;
                    for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                        for (int j = 0; j < 4; j++)
                            if (COMM_SUCCESS != packetHandlerRightArm->write2ByteTxRx(portHandlerRightArm, i, IND_ADDR_GOAL_POSITION + (2 * j), ADDR_GOAL_POSITION + j, &dxl_error))
                                return false;
                    for (int i = BODY_ID_MIN; i <= BODY_ID_MAX; i++)
                        for (int j = 0; j < 4; j++)
                            if (COMM_SUCCESS != packetHandlerBody->write2ByteTxRx(portHandlerBody, i, IND_ADDR_GOAL_POSITION + (2 * j), ADDR_GOAL_POSITION + j, &dxl_error))
                                return false;
                }


                // Goal Current
                for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++) {
                    for (int j = 0; j < 2; j++) {
                        if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerHead, i, IND_ADDR_GOAL_CURRENT + (2 * j), ADDR_GOAL_CURRENT + j, &dxl_error)) {
                            std::cout << "Goal current not set for head servo ID: " << i << ", byte: " << j << std::endl;
                            return false;
                        }
                    }
                }
                if (EpiFullMode)
                {
                    for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                        for (int j = 0; j < 2; j++)
                            if (COMM_SUCCESS != packetHandlerLeftArm->write2ByteTxRx(portHandlerLeftArm, i, IND_ADDR_GOAL_CURRENT + (2 * j), ADDR_GOAL_CURRENT + j, &dxl_error))
                                return false;
                    for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                        for (int j = 0; j < 2; j++)
                            if (COMM_SUCCESS != packetHandlerRightArm->write2ByteTxRx(portHandlerRightArm, i, IND_ADDR_GOAL_CURRENT + (2 * j), ADDR_GOAL_CURRENT + j, &dxl_error))
                                return false;
                    for (int i = BODY_ID_MIN; i <= BODY_ID_MAX; i++)
                        for (int j = 0; j < 2; j++)
                            if (COMM_SUCCESS != packetHandlerBody->write2ByteTxRx(portHandlerBody, i, IND_ADDR_GOAL_CURRENT + (2 * j), ADDR_GOAL_CURRENT + j, &dxl_error))
                                return false;
                }

                // Goal PWM
                for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++) {
                    for (int j = 0; j < 2; j++) {
                        if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerHead, i, IND_ADDR_GOAL_PWM + (2 * j), ADDR_GOAL_PWM + j, &dxl_error)) {
                            std::cout << "Goal PWM not set for head servo ID: " << i << ", byte: " << j << std::endl;
                            return false;
                        }
                    }
                }
                if(EpiFullMode){
                    for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                        for (int j = 0; j < 2; j++)
                            if (COMM_SUCCESS != packetHandlerLeftArm->write2ByteTxRx(portHandlerLeftArm, i, IND_ADDR_GOAL_PWM + (2 * j), ADDR_GOAL_PWM + j, &dxl_error))
                                return false;
                }

                // Indirect adress (present position). Feedback
                for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++){
                    for (int j = 0; j < 4; j++){
                        if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerHead, i, IND_ADDR_PRESENT_POSITION + (2 * j), ADDR_PRESENT_POSITION + j, &dxl_error)){
                            std::cout << "Present position not set for head servo ID: " << i << ", byte: " << j << std::endl;
                            return false;
                        }
                    }
                }
                if (EpiFullMode)
                {
                    for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                        for (int j = 0; j < 4; j++)
                            if (COMM_SUCCESS != packetHandlerLeftArm->write2ByteTxRx(portHandlerLeftArm, i, IND_ADDR_PRESENT_POSITION + (2 * j), ADDR_PRESENT_POSITION + j, &dxl_error))
                                return false;
                    for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                        for (int j = 0; j < 4; j++)
                            if (COMM_SUCCESS != packetHandlerRightArm->write2ByteTxRx(portHandlerRightArm, i, IND_ADDR_PRESENT_POSITION + (2 * j), ADDR_PRESENT_POSITION + j, &dxl_error))
                                return false;
                    for (int i = BODY_ID_MIN; i <= BODY_ID_MAX; i++)
                        for (int j = 0; j < 4; j++)
                            if (COMM_SUCCESS != packetHandlerBody->write2ByteTxRx(portHandlerBody, i, IND_ADDR_PRESENT_POSITION + (2 * j), ADDR_PRESENT_POSITION + j, &dxl_error))
                                return false;
                }
                // Indirect adress (present current). Feedback. MX28 does not support current mode.
                for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++){
                    for (int j = 0; j < 2; j++){
                        if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerHead, i, IND_ADDR_PRESENT_CURRENT + (2 * j), ADDR_PRESENT_CURRENT + j, &dxl_error)){
                            std::cout << "Present current not set for head servo ID: " << i << ", byte: " << j << std::endl;
                            return false;
                        }
                    }
                }
                if (EpiFullMode)
                {
                    for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                        for (int j = 0; j < 2; j++)
                            if (COMM_SUCCESS != packetHandlerLeftArm->write2ByteTxRx(portHandlerLeftArm, i, IND_ADDR_PRESENT_CURRENT + (2 * j), ADDR_PRESENT_CURRENT + j, &dxl_error))
                                return false;
                    for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                        for (int j = 0; j < 2; j++)
                            if (COMM_SUCCESS != packetHandlerRightArm->write2ByteTxRx(portHandlerRightArm, i, IND_ADDR_PRESENT_CURRENT + (2 * j), ADDR_PRESENT_CURRENT + j, &dxl_error))
                                return false;
                    for (int i = BODY_ID_MIN; i <= BODY_ID_MAX; i++)
                        for (int j = 0; j < 2; j++)
                            if (COMM_SUCCESS != packetHandlerBody->write2ByteTxRx(portHandlerBody, i, IND_ADDR_PRESENT_CURRENT + (2 * j), ADDR_PRESENT_CURRENT + j, &dxl_error))
                                return false;
                }
                // Indirect adress (present temperature).
                for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++){
                    if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerHead, i, IND_ADDR_PRESENT_TEMPERATURE, ADDR_PRESENT_TEMPERATURE, &dxl_error)){
                        std::cout << "Present temperature indir not set for head servo ID: " << i << ", byte: " << std::endl;
                        return false;
                    }
                }
                if (EpiFullMode)
                {
                    for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                        if (COMM_SUCCESS != packetHandlerLeftArm->write2ByteTxRx(portHandlerLeftArm, i, IND_ADDR_PRESENT_TEMPERATURE, ADDR_PRESENT_TEMPERATURE, &dxl_error))
                            return false;
                    for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                        if (COMM_SUCCESS != packetHandlerRightArm->write2ByteTxRx(portHandlerRightArm, i, IND_ADDR_PRESENT_TEMPERATURE, ADDR_PRESENT_TEMPERATURE, &dxl_error))
                            return false;
                    for (int i = BODY_ID_MIN; i <= BODY_ID_MAX; i++)
                        if (COMM_SUCCESS != packetHandlerBody->write2ByteTxRx(portHandlerBody, i, IND_ADDR_PRESENT_TEMPERATURE, ADDR_PRESENT_TEMPERATURE, &dxl_error))
                            return false;
                }  
                
            }

            //Write settings from json file to the servos            
            for (int id = idMin; id <= idMax; id++) {
                printf("Setting control table for servo ID: %d\n", id);
                for (int param = 0; param < parameter_lst.size(); param++) {
                    parameterName = std::string(parameter_lst[param]);
                    byteLength = servoControlTable[parameterName]["Bytes"];
                    for (int byte = 0; byte < byteLength; byte++) {
                        directAddress = static_cast<int>(servoControlTable[parameterName]["Address"]) + byte;    
                        //Writing settings to the servos
                        if (!parameterName.equals("Present Current") && !parameterName.equals("Present Position")) { // Present current and present position is not used for writing
                            // 2 bytes parameters
                            if (byteLength==2){
                                param_default_2Byte = data(id-2, param);
                                if(parameterName.equals("Goal PWM"))
                                    param_default_2Byte =  data(id-2, param)/0.11299;
                                if (COMM_SUCCESS != packetHandlers[p]->write2ByteTxRx(portHandlers[p], id, directAddress, param_default_2Byte, &dxl_error)) {
                                    std::cout << "Failed to set " << parameterName
                                            << " for servo ID: " << id
                                            << " of port:" << portHandlers[p]->getPortName()
                                            << " Error: " << packetHandlers[p]->getTxRxResult(dxl_comm_result)
                                            << " DXL Error: " << packetHandlers[p]->getRxPacketError(dxl_error) << std::endl;
                                    return false;
                                }
                            }
                            if (byteLength==4){
                                param_default_4Byte = data(id-2, param);
                                if (COMM_SUCCESS != packetHandlers[p]->write4ByteTxRx(portHandlers[p], id, directAddress, param_default_4Byte, &dxl_error)) {
                                    std::cout << "Failed to set " << parameterName
                                            << " for servo ID: " << id
                                            << " of port:" << portHandlers[p]->getPortName()
                                            << " Error: " << packetHandlers[p]->getTxRxResult(dxl_comm_result)
                                            << " DXL Error: " << packetHandlers[p]->getRxPacketError(dxl_error) << std::endl;
                                    return false;
                                }
                            }
                            
                        }
                        else {// Only present position and present current is used for reading
                            // std::cout << "Setting indirect reading address for " 
                            // << parameterName << " indirect address: "
                            // << addressRead << " direct address: " 
                            // << directAddress <<std::endl;
                            // Reading Indirect Addresses
                            
                            if (COMM_SUCCESS != packetHandlers[p]->write2ByteTxRx(portHandlers[p], id, addressRead, directAddress, &dxl_error)) {
                                std::cout << "Failed to set indirect reading address for " << parameterName
                                        << " for servo ID: " << id
                                        << " of port:" << portHandlers[p]->getPortName()
                                        << " Error: " << packetHandlers[p]->getTxRxResult(dxl_comm_result)
                                        << " DXL Error: " << packetHandlers[p]->getRxPacketError(dxl_error) << std::endl;
                                return false;
                            }

                            // Increment address after each byte
                            addressRead += 2;

                        }
                       

                        
                        
                    }//for byte
                }
            }//for id
            
            
           
            
        }//for portHandlers
      
        


        return true; // Yay we manage to set everything we needed.
    }

    bool SetPupilParameters(){
        Notify(msg_debug, "Settting parameters for pupils (servo XL320)\n");
        uint16_t pupil_moving_speed = 150;
        uint8_t pupil_p_gain = 100;
        uint8_t pupil_i_gain = 20;
        uint8_t pupil_d_gain = 5;
        uint8_t dxl_error = 0;

        Timer t;
        double xlTimer = 0.01; // Timer in sec. XL320 need this. Not sure why.

        // PUPIL ID 2 (Left pupil)
        // Limit position min
        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 2, 6, AngleMinLimitPupil[0], &dxl_error))
            return false;
        Sleep(xlTimer);
        // Limit position max
        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 2, 8, AngleMaxLimitPupil[0], &dxl_error))
            return false;
        Sleep(xlTimer);

        // Moving speed
        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 2, 32, pupil_moving_speed, &dxl_error))
            return false;
        Sleep(xlTimer);

        // P
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 2, 29, pupil_p_gain, &dxl_error))
            return false;
        Sleep(xlTimer);


        // I
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 2, 28, pupil_i_gain, &dxl_error))
            return false;
        Sleep(xlTimer);


        // D
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 2, 27, pupil_d_gain, &dxl_error))
            return false;
        Sleep(xlTimer);


        // PUPIL ID 3 (Right pupil)
        // Limit position in
        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 3, 6, AngleMinLimitPupil[1], &dxl_error))
            return false;
        Sleep(xlTimer);


        // Limit position max
        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 3, 8, AngleMaxLimitPupil[1], &dxl_error))
            return false;
        Sleep(xlTimer);


        // Moving speed
        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 3, 32, pupil_moving_speed, &dxl_error))
            return false;
        Sleep(xlTimer);

        // P
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 3, 29, pupil_p_gain, &dxl_error))
            return false;
        Sleep(xlTimer);


        // I
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 3, 28, pupil_i_gain, &dxl_error))
            return false;
        Sleep(xlTimer);


        // D
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 3, 27, pupil_d_gain, &dxl_error))
            return false;
        Sleep(xlTimer);


        std::cout << "Pupil parameters set" << std::endl;

        Notify(msg_debug, "Pupil parameters done\n");

        return true;
    }

    bool SetMinMaxLimits(){
        uint32_t param_default_4Byte;
        uint8_t dxl_error = 0;
        int idMin;
        int idMax;
    
        std::vector<int> maxMinPositionLimitIndex;
        //min and max limits
        for (int p = 0; p < portHandlers.size(); p++) {

            // Ensure p is within the valid range
            if (p < 0 || p >= portHandlers.size() || p >= packetHandlers.size()) {
                std::cout << "Invalid index for portHandlers or packetHandlers: " << p << std::endl;
                return false;
            }

            // Ensure pointers are not null
            if (!portHandlers[p] || !packetHandlers[p]) {
                std::cout << "Null pointer encountered in portHandlers or packetHandlers at index: " << p << std::endl;
                return false;
            }
            //switch statement for different p values. 
            switch (p) {
                case 0:
                maxMinPositionLimitIndex = {0, 1, 2, 3};
                idMin = HEAD_ID_MIN;
                idMax = HEAD_ID_MAX;
                break;
                //same for p==1 and p==2 (left and right arm)
                case 1:
                maxMinPositionLimitIndex = {4, 5, 6, 7, 8, 9};
                idMin = ARM_ID_MIN;
                idMax = ARM_ID_MAX;
                break;
                case 2:
                maxMinPositionLimitIndex = {10, 11, 12, 13, 14, 15};
                idMin = ARM_ID_MIN;
                idMax = ARM_ID_MAX;
                break;
                case 3:
                maxMinPositionLimitIndex = {16};
                idMin = BODY_ID_MIN;
                idMax = BODY_ID_MAX;
                break;
                }
            int i = 0;
            for (int id = idMin; id <= idMax; id++) 
            {
                //min and max limits
                param_default_4Byte = minLimitPosition[maxMinPositionLimitIndex[i]] /360 * 4096;
  
                if (COMM_SUCCESS != packetHandlers[p]->write4ByteTxRx(portHandlers[p], id, ADDR_MIN_POSITION_LIMIT, param_default_4Byte, &dxl_error)){
                    std::cout << "Failed to set indirect address for min position limit for servo ID: " 
                    << id << " of port:" 
                    << portHandlers[p]->getPortName() 
                    << ", DXL Error: " << packetHandlers[p]->getRxPacketError(dxl_error) << std::endl;
                    
                    return false;
                }
                param_default_4Byte = maxLimitPosition[maxMinPositionLimitIndex[i]]/ 360.0 * 4096.0;
                
                if (COMM_SUCCESS != packetHandlers[p]->write4ByteTxRx(portHandlers[p], id, ADDR_MAX_POSITION_LIMIT, param_default_4Byte, &dxl_error)){ 
                    std::cout << "Failed to set indirect address for max position limit for servo ID: " 
                    << id << " of port:" 
                    << portHandlers[p]->getPortName() 
                    << ", DXL Error: " << packetHandlers[p]->getRxPacketError(dxl_error) << std::endl;
                
                    return false;
                }
                i++;
            }
        }
        return true;
    }
    

    bool SetDefaultServoSettings() {
        
        uint32_t param_default_4Byte;
        uint32_t profile_acceleration = 0;
        uint32_t profile_velocity = 0;
        
        uint16_t p_gain_head = 850;
        uint16_t i_gain_head = 0;
        uint16_t d_gain_head = 0;
        uint16_t goal_pwm = 100;
        
        uint16_t p_gain_arm = 100;
        uint16_t i_gain_arm = 0;
        uint16_t d_gain_arm = 1000;
        
        uint16_t p_gain_body = 100;
        uint16_t i_gain_body = 0;
        uint16_t d_gain_body = 1000;

       
        uint16_t pupil_moving_speed = 150;
        uint8_t param_default_1Byte;
        uint8_t pupil_p_gain = 100;
        uint8_t pupil_i_gain = 20;
        uint8_t pupil_d_gain = 5;


        uint8_t dxl_error = 0; 
        int dxl_comm_result = COMM_TX_FAIL;

        Notify(msg_debug, "Setting control table on servos\n");

        // Torque Enable
        for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++) {
            if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerHead, i, IND_ADDR_TORQUE_ENABLE, ADDR_TORQUE_ENABLE, &dxl_error)) {
                std::cout << "Failed to set torque enable for head servo ID: " << i << std::endl;
                return false;
            }
        }
        if (EpiFullMode) {
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++) {
                if (COMM_SUCCESS != packetHandlerLeftArm->write2ByteTxRx(portHandlerLeftArm, i, IND_ADDR_TORQUE_ENABLE, ADDR_TORQUE_ENABLE, &dxl_error)) {
                    std::cout << "Failed to set torque enable for left arm servo ID: " << i << std::endl;
                    return false;
                }
            }
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++) {
                if (COMM_SUCCESS != packetHandlerRightArm->write2ByteTxRx(portHandlerRightArm, i, IND_ADDR_TORQUE_ENABLE, ADDR_TORQUE_ENABLE, &dxl_error)) {
                    std::cout << "Failed to set torque enable for right arm servo ID: " << i << std::endl;
                    return false;
                }
            }
            for (int i = BODY_ID_MIN; i <= BODY_ID_MAX; i++) {
                if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerBody, i, IND_ADDR_TORQUE_ENABLE, ADDR_TORQUE_ENABLE, &dxl_error)) {
                    std::cout << "Failed to set torque enable for body servo ID: " << i << std::endl;
                    return false;
                }
            }
        } 

        // Goal Position
        for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++) {
            for (int j = 0; j < 4; j++) {
                if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerHead, i, IND_ADDR_GOAL_POSITION + (2 * j), ADDR_GOAL_POSITION + j, &dxl_error)) {
                    std::cout << "Failed to set goal position for head servo ID: " << i << ", byte: " << j << std::endl;
                    return false;
                }
            }
        }
         if (EpiFullMode)
        {
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                for (int j = 0; j < 4; j++)
                    if (COMM_SUCCESS != packetHandlerLeftArm->write2ByteTxRx(portHandlerLeftArm, i, IND_ADDR_GOAL_POSITION + (2 * j), ADDR_GOAL_POSITION + j, &dxl_error))
                        return false;
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                for (int j = 0; j < 4; j++)
                    if (COMM_SUCCESS != packetHandlerRightArm->write2ByteTxRx(portHandlerRightArm, i, IND_ADDR_GOAL_POSITION + (2 * j), ADDR_GOAL_POSITION + j, &dxl_error))
                        return false;
            for (int i = BODY_ID_MIN; i <= BODY_ID_MAX; i++)
                for (int j = 0; j < 4; j++)
                    if (COMM_SUCCESS != packetHandlerBody->write2ByteTxRx(portHandlerBody, i, IND_ADDR_GOAL_POSITION + (2 * j), ADDR_GOAL_POSITION + j, &dxl_error))
                        return false;
        }


        // Goal Current
        for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++) {
            for (int j = 0; j < 2; j++) {
                if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerHead, i, IND_ADDR_GOAL_CURRENT + (2 * j), ADDR_GOAL_CURRENT + j, &dxl_error)) {
                    std::cout << "Goal current not set for head servo ID: " << i << ", byte: " << j << std::endl;
                    return false;
                }
            }
        }
         // Indirect adress (present position). Feedback
        for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++){
            for (int j = 0; j < 4; j++){
                if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerHead, i, IND_ADDR_PRESENT_POSITION + (2 * j), ADDR_PRESENT_POSITION + j, &dxl_error)){
                    std::cout << "Present position not set for head servo ID: " << i << ", byte: " << j << std::endl;
                    return false;
                }
            }
        }
        if (EpiFullMode)
        {
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                for (int j = 0; j < 4; j++)
                    if (COMM_SUCCESS != packetHandlerLeftArm->write2ByteTxRx(portHandlerLeftArm, i, IND_ADDR_PRESENT_POSITION + (2 * j), ADDR_PRESENT_POSITION + j, &dxl_error))
                        return false;
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                for (int j = 0; j < 4; j++)
                    if (COMM_SUCCESS != packetHandlerRightArm->write2ByteTxRx(portHandlerRightArm, i, IND_ADDR_PRESENT_POSITION + (2 * j), ADDR_PRESENT_POSITION + j, &dxl_error))
                        return false;
            for (int i = BODY_ID_MIN; i <= BODY_ID_MAX; i++)
                for (int j = 0; j < 4; j++)
                    if (COMM_SUCCESS != packetHandlerBody->write2ByteTxRx(portHandlerBody, i, IND_ADDR_PRESENT_POSITION + (2 * j), ADDR_PRESENT_POSITION + j, &dxl_error))
                        return false;
        }
        // Indirect adress (present current). Feedback. MX28 does not support current mode.
        for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++){
            for (int j = 0; j < 2; j++){
                if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerHead, i, IND_ADDR_PRESENT_CURRENT + (2 * j), ADDR_PRESENT_CURRENT + j, &dxl_error)){
                    std::cout << "Present current not set for head servo ID: " << i << ", byte: " << j << std::endl;
                    return false;
                }
            }
        }
        if (EpiFullMode)
        {
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                for (int j = 0; j < 2; j++)
                    if (COMM_SUCCESS != packetHandlerLeftArm->write2ByteTxRx(portHandlerLeftArm, i, IND_ADDR_PRESENT_CURRENT + (2 * j), ADDR_PRESENT_CURRENT + j, &dxl_error))
                        return false;
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                for (int j = 0; j < 2; j++)
                    if (COMM_SUCCESS != packetHandlerRightArm->write2ByteTxRx(portHandlerRightArm, i, IND_ADDR_PRESENT_CURRENT + (2 * j), ADDR_PRESENT_CURRENT + j, &dxl_error))
                        return false;
            for (int i = BODY_ID_MIN; i <= BODY_ID_MAX; i++)
                for (int j = 0; j < 2; j++)
                    if (COMM_SUCCESS != packetHandlerBody->write2ByteTxRx(portHandlerBody, i, IND_ADDR_PRESENT_CURRENT + (2 * j), ADDR_PRESENT_CURRENT + j, &dxl_error))
                        return false;
        }
        // Indirect adress (present temperature).
        for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++){
            if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerHead, i, IND_ADDR_PRESENT_TEMPERATURE, ADDR_PRESENT_TEMPERATURE, &dxl_error)){
                std::cout << "Present temperature indir not set for head servo ID: " << i << ", byte: " << std::endl;
                return false;
            }
        }
        if (EpiFullMode)
        {
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerLeftArm->write2ByteTxRx(portHandlerLeftArm, i, IND_ADDR_PRESENT_TEMPERATURE, ADDR_PRESENT_TEMPERATURE, &dxl_error))
                    return false;
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerRightArm->write2ByteTxRx(portHandlerRightArm, i, IND_ADDR_PRESENT_TEMPERATURE, ADDR_PRESENT_TEMPERATURE, &dxl_error))
                    return false;
            for (int i = BODY_ID_MIN; i <= BODY_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerBody->write2ByteTxRx(portHandlerBody, i, IND_ADDR_PRESENT_TEMPERATURE, ADDR_PRESENT_TEMPERATURE, &dxl_error))
                    return false;
        }

        // Profile acceleration

    for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++){
            if (COMM_SUCCESS != packetHandlerHead->write4ByteTxRx(portHandlerHead, i, ADDR_PROFILE_ACCELERATION, profile_acceleration, &dxl_error)){
                std::cout << "Profile acceleration for head servo ID: " << i << std::endl;
                return false;
            }
        }
        if (EpiFullMode)
        {
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, i, ADDR_PROFILE_ACCELERATION, profile_acceleration, &dxl_error))
                    return false;
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, i, ADDR_PROFILE_ACCELERATION, profile_acceleration, &dxl_error))
                    return false;
            for (int i = BODY_ID_MIN; i <= BODY_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerBody->write4ByteTxRx(portHandlerBody, i, ADDR_PROFILE_ACCELERATION, profile_acceleration, &dxl_error))
                    return false;
        }

        // Common settings for the servos
        // Profile velocity (210)

        for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++){
            if (COMM_SUCCESS != packetHandlerHead->write4ByteTxRx(portHandlerHead, i, ADDR_PROFILE_VELOCITY, profile_velocity, &dxl_error)){
                std::cout << "Profile velocity not set for head servo ID: " << i << std::endl;
                return false;
                }
        }
        if (EpiFullMode)
        {
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, i, ADDR_PROFILE_VELOCITY, profile_velocity, &dxl_error))
                    return false;
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, i, ADDR_PROFILE_VELOCITY, profile_velocity, &dxl_error))
                    return false;
            for (int i = BODY_ID_MIN; i <= BODY_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerBody->write4ByteTxRx(portHandlerBody, i, ADDR_PROFILE_VELOCITY, profile_velocity, &dxl_error))
                    return false;
        }

        // P (100)
        for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++){
            if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerHead, i, ADDR_P, p_gain_head, &dxl_error)){
                std::cout << "P (PID) not set for head servo ID: " << i <<std::endl;
                return false;
            }
        }
        if (EpiFullMode)
        {
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerLeftArm->write2ByteTxRx(portHandlerLeftArm, i, ADDR_P, p_gain_arm, &dxl_error))
                    return false;
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerRightArm->write2ByteTxRx(portHandlerRightArm, i, ADDR_P, p_gain_arm, &dxl_error))
                    return false;
            for (int i = BODY_ID_MIN; i <= BODY_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerBody->write2ByteTxRx(portHandlerBody, i, ADDR_P, p_gain_body, &dxl_error))
                    return false;
        }

        // I
        // The I value almost killed Epi.
        for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++){
            if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerHead, i, ADDR_I, i_gain_head, &dxl_error)){
                std::cout << "I (PID) not set for head servo ID: " << i<< std::endl;
                return false;
            }
        }
        if (EpiFullMode)
        {
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerLeftArm->write2ByteTxRx(portHandlerLeftArm, i, ADDR_I, i_gain_arm, &dxl_error))
                    return false;
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerRightArm->write2ByteTxRx(portHandlerRightArm, i, ADDR_I, i_gain_arm, &dxl_error))
                    return false;
            for (int i = BODY_ID_MIN; i <= BODY_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerBody->write2ByteTxRx(portHandlerBody, i, ADDR_I, i_gain_body, &dxl_error))
                    return false;
        }

        // D
        for (int i = HEAD_ID_MIN; i <= HEAD_ID_MAX; i++){
            if (COMM_SUCCESS != packetHandlerHead->write2ByteTxRx(portHandlerHead, i, ADDR_D, d_gain_head, &dxl_error)){
                std::cout << "D (PID) not set for head servo ID: " << i << std::endl;
                return false;
            }
        }
        if (EpiFullMode)
        {
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerLeftArm->write2ByteTxRx(portHandlerLeftArm, i, ADDR_D, d_gain_arm, &dxl_error))
                    return false;
            for (int i = ARM_ID_MIN; i <= ARM_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerRightArm->write2ByteTxRx(portHandlerRightArm, i, ADDR_D, d_gain_arm, &dxl_error))
                    return false;
            for (int i = BODY_ID_MIN; i <= BODY_ID_MAX; i++)
                if (COMM_SUCCESS != packetHandlerBody->write2ByteTxRx(portHandlerBody, i, ADDR_D, d_gain_body, &dxl_error))
                    return false;
        }

        // Specific setting for each servos
        // HEAD ID 2
        // Limit position max
        uint32_t limit_pos_max_tilt = 2700;
        if (COMM_SUCCESS != packetHandlerHead->write4ByteTxRx(portHandlerHead, 2, 48, limit_pos_max_tilt, &dxl_error)){
            std::cout << "Max limit not set for head servo ID: 2 " << std::endl;
            return false;
            }
        // Limit position min
        uint32_t limit_pos_min_tilt = 1300;
        if (COMM_SUCCESS != packetHandlerHead->write4ByteTxRx(portHandlerHead, 2, 52, limit_pos_min_tilt, &dxl_error)){
            std::cout << "Min limit not set for head servo ID: 2 " << std::endl;
            return false;
            }

        // HEAD ID 3
        // Limit position max
        uint32_t limit_pos_max_pan = 2500;
        if (COMM_SUCCESS != packetHandlerHead->write4ByteTxRx(portHandlerHead, 3, 48, limit_pos_max_pan, &dxl_error))
            return false;
        // Limit position min
        uint32_t limit_pos_min_pan = 1750;
        if (COMM_SUCCESS != packetHandlerHead->write4ByteTxRx(portHandlerHead, 3, 52, limit_pos_min_pan, &dxl_error))
            return false;

        // HEAD ID 4 (Left eye)
        // Limit position max
        uint32_t limit_pos_max_left_eye = 2300;
        if (COMM_SUCCESS != packetHandlerHead->write4ByteTxRx(portHandlerHead, 4, 48, limit_pos_max_left_eye, &dxl_error))
            return false;
        // Limit position min
        uint32_t limit_pos_min_left_eye = 1830;
        if (COMM_SUCCESS != packetHandlerHead->write4ByteTxRx(portHandlerHead, 4, 52, limit_pos_min_left_eye, &dxl_error))
            return false;

        // HEAD ID 5 (Right eye)
        // Limit position max
        uint32_t limit_pos_max_right_eye = 2200;
        if (COMM_SUCCESS != packetHandlerHead->write4ByteTxRx(portHandlerHead, 5, 48, limit_pos_max_right_eye, &dxl_error))
            return false;

        // Limit position min
        uint32_t limit_pos_min_right_eye = 1780;
        if (COMM_SUCCESS != packetHandlerHead->write4ByteTxRx(portHandlerHead, 5, 52, limit_pos_min_right_eye, &dxl_error))
            return false;

        Timer t;
        double xlTimer = 0.01; // Timer in sec. XL320 need this. Not sure why.

        // PUPIL ID 2 (Left pupil)
        // Limit position min
        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 2, 6, AngleMinLimitPupil[0], &dxl_error))
            return false;
        Sleep(xlTimer);

        // Limit position max
        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 2, 8, AngleMaxLimitPupil[0], &dxl_error))
            return false;
        Sleep(xlTimer);

        // Moving speed
        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 2, 32, pupil_moving_speed, &dxl_error))
            return false;
        Sleep(xlTimer);

        // P
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 2, 29, pupil_p_gain, &dxl_error))
            return false;
        Sleep(xlTimer);


        // I
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 2, 28, pupil_i_gain, &dxl_error))
            return false;
        Sleep(xlTimer);


        // D
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 2, 27, pupil_d_gain, &dxl_error))
            return false;
        Sleep(xlTimer);


        // PUPIL ID 3 (Right pupil)
        // Limit position in
        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 3, 6, AngleMinLimitPupil[1], &dxl_error))
            return false;
        Sleep(xlTimer);


        // Limit position max
        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 3, 8, AngleMaxLimitPupil[1], &dxl_error))
            return false;
        Sleep(xlTimer);


        // Moving speed
        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 3, 32, pupil_moving_speed, &dxl_error))
            return false;
        Sleep(xlTimer);

        // P
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 3, 29, pupil_p_gain, &dxl_error))
            return false;
        Sleep(xlTimer);


        // I
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 3, 28, pupil_i_gain, &dxl_error))
            return false;
        Sleep(xlTimer);


        // D
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 3, 27, pupil_d_gain, &dxl_error))
            return false;
        Sleep(xlTimer);


        if (EpiFullMode)
        {
            // LEFT ARM ID 2
            // Limit position max
            param_default_4Byte = 3200;
            if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, 2, 48, param_default_4Byte, &dxl_error))
                return false;
            // Limit position min
            param_default_4Byte = 600;
            if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, 2, 52, param_default_4Byte, &dxl_error))
                return false;

            // LEFT ARM ID 3
            // Limit position max
            param_default_4Byte = 3200;
            if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, 3, 48, param_default_4Byte, &dxl_error))
                return false;
            // Limit position min
            param_default_4Byte = 800;
            if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, 3, 52, param_default_4Byte, &dxl_error))
                return false;

            // LEFT ARM ID 4
            // Limit position max
            param_default_4Byte = 3000;
            if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, 4, 48, param_default_4Byte, &dxl_error))
                return false;
            // Limit position min
            param_default_4Byte = 1000;
            if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, 4, 52, param_default_4Byte, &dxl_error))
                return false;

            // LEFT ARM ID 5
            // Limit position max
            param_default_4Byte = 2300;
            if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, 5, 48, param_default_4Byte, &dxl_error))
                return false;
            // Limit position min
            param_default_4Byte = 600;
            if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, 5, 52, param_default_4Byte, &dxl_error))
                return false;

            // LEFT ARM ID 6
            // Limit position max
            param_default_4Byte = 3900;
            if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, 6, 48, param_default_4Byte, &dxl_error))
                return false;
            // Limit position min
            param_default_4Byte = 800;
            if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, 6, 52, param_default_4Byte, &dxl_error))
                return false;

            // LEFT ARM ID 7
            // Limit position max
            param_default_4Byte = 4095;
            if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, 7, 48, param_default_4Byte, &dxl_error))
                return false;
            // Limit position min
            param_default_4Byte = 0;
            if (COMM_SUCCESS != packetHandlerLeftArm->write4ByteTxRx(portHandlerLeftArm, 7, 52, param_default_4Byte, &dxl_error))
                return false;

            // RIGHT ARM ID 2
            // Limit position max
            param_default_4Byte = 3200;
            if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, 2, 48, param_default_4Byte, &dxl_error))
                return false;
            // Limit position min
            param_default_4Byte = 600;
            if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, 2, 52, param_default_4Byte, &dxl_error))
                return false;
            // RIGHT ARM ID 3
            // Limit position max
            param_default_4Byte = 3300;
            if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, 3, 48, param_default_4Byte, &dxl_error))
                return false;
            // Limit position min
            param_default_4Byte = 900;
            if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, 3, 52, param_default_4Byte, &dxl_error))
                return false;
            // RIGHT ARM ID 4
            // Limit position max
            param_default_4Byte = 3000;
            if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, 4, 48, param_default_4Byte, &dxl_error))
                return false;
            // Limit position min
            param_default_4Byte = 1000;
            if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, 4, 52, param_default_4Byte, &dxl_error))
                return false;
            // RIGHT ARM ID 5
            // Limit position max
            param_default_4Byte = 3600;
            if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, 5, 48, param_default_4Byte, &dxl_error))
                return false;
            // Limit position min
            param_default_4Byte = 1800;
            if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, 5, 52, param_default_4Byte, &dxl_error))
                return false;
            // RIGHT ARM ID 6
            // Limit position max
            param_default_4Byte = 3900;
            if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, 6, 48, param_default_4Byte, &dxl_error))
                return false;
            // Limit position min
            param_default_4Byte = 800;
            if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, 6, 52, param_default_4Byte, &dxl_error))
                return false;
            // RIGHT ARM ID 7
            // Limit position max
            param_default_4Byte = 4095;
            if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, 7, 48, param_default_4Byte, &dxl_error))
                return false;
            // Limit position min
            param_default_4Byte = 0;
            if (COMM_SUCCESS != packetHandlerRightArm->write4ByteTxRx(portHandlerRightArm, 7, 52, param_default_4Byte, &dxl_error))
                return false;

            // BODY ID 2
            // Limit position max
            param_default_4Byte = 3900;
            if (COMM_SUCCESS != packetHandlerBody->write4ByteTxRx(portHandlerBody, 2, 48, param_default_4Byte, &dxl_error))
                return false;
            // Limit position min
            param_default_4Byte = 100;
            if (COMM_SUCCESS != packetHandlerBody->write4ByteTxRx(portHandlerBody, 2, 52, param_default_4Byte, &dxl_error))
                return false;
        }
        return true; // Yay we manage to set everything we needed.
    }





    bool PowerOn(int IDMin, int IDMax, dynamixel::PortHandler *portHandler, dynamixel::PacketHandler *packetHandler)
    {
        if (portHandler == NULL) // If no port handler return true. Only return false if communication went wrong.
            return true;

        Debug("Power on servos");

        Timer t;
        const int nrOfServos = IDMax - IDMin + 1;
        int dxl_comm_result = COMM_TX_FAIL; // Communication result
        uint8_t dxl_error = 0;              // Dynamixel error
        uint16_t start_p_value[7] = {0, 0, 0, 0, 0, 0, 0};
        uint32_t present_postition_value[7] = {0, 0, 0, 0, 0, 0, 0};

        // Set torque value to 0
        for (int i = 0; i < nrOfServos; i++) {
            if (COMM_SUCCESS != packetHandler->write1ByteTxRx(portHandler, IDMin + i, 64, 0, &dxl_error)) {
            Warning("Can not turn off torque for servo ID: " + std::to_string(IDMin + i));
            return false;
            }
        }

        // Get P values
        for (int i = 0; i < nrOfServos; i++) {
            if (COMM_SUCCESS != packetHandler->read2ByteTxRx(portHandler, IDMin + i, 84, &start_p_value[i], &dxl_error)) {
            Warning("Can not read P value for servo ID: " + std::to_string(IDMin + i));
            return false;
            }
        }

        // Set P value to 0
        for (int i = 0; i < nrOfServos; i++) {
            if (COMM_SUCCESS != packetHandler->write2ByteTxRx(portHandler, IDMin + i, 84, 0, &dxl_error)) {
            Warning("Can not set P value to 0 for servo ID: " + std::to_string(IDMin + i));
            return false;
            }
        }

        // Set torque value to 1
        for (int i = 0; i < nrOfServos; i++) {
            if (COMM_SUCCESS != packetHandler->write1ByteTxRx(portHandler, IDMin + i, 64, 1, &dxl_error)) {
            Warning("Can not turn on torque for servo ID: " + std::to_string(IDMin + i));
            return false;
            }
        }
        
        while (t.GetTime() < TIMER_POWER_ON) {
            // Get present position
            for (int i = 0; i < nrOfServos; i++) {
            if (COMM_SUCCESS != packetHandler->read4ByteTxRx(portHandler, IDMin + i, 132, &present_postition_value[i], &dxl_error)) {
                Warning("Can not read present position for servo ID: " + std::to_string(IDMin + i));
                return false;
            }
            }
            
            // Set goal position to present postiion
            for (int i = 0; i < nrOfServos; i++) {
            if (COMM_SUCCESS != packetHandler->write4ByteTxRx(portHandler, IDMin + i, 116, present_postition_value[i], &dxl_error)) {
                Warning("Can not set goal position for servo ID: " + std::to_string(IDMin + i));
                return false;
            }
            }
            
            Sleep(0.01); // Sleep for 10 ms to allow the servos to move to the goal position.
            // Ramping up P
            for (int i = 0; i < nrOfServos; i++) {
            if (COMM_SUCCESS != packetHandler->write2ByteTxRx(portHandler, IDMin + i, 84, int(float(start_p_value[i]) / float(TIMER_POWER_ON) * t.GetTime()), &dxl_error)) {
                Warning("Can not ramp up P value for servo ID: " + std::to_string(IDMin + i));
                return false;
            }
            }
        }

        // Set P value to start value
        for (int i = 0; i < nrOfServos; i++) {
            if (COMM_SUCCESS != packetHandler->write2ByteTxRx(portHandler, IDMin + i, 84, start_p_value[i], &dxl_error)) {
            Warning("Can not restore P value for servo ID: " + std::to_string(IDMin + i));
            return false;
            }
        }

        return true;
    }

    bool PowerOnPupil()
    {
        uint8_t dxl_error = 0; // Dynamixel error

        // Enable torque. No fancy rampiong
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 2, 24, 1, &dxl_error))
            return false;
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 3, 24, 1, &dxl_error))
            return false;
        return true;
    }

    bool PowerOnRobot()
    {
        // Trying to torque up the power of the servos.
        // Dynamixel protocel 2.0
        // In current base position control mode goal current can be used.
        // In position control mode P can be used (PID).
        // Torqing up the servos? This can not be done in 2.0 and position mode only in position-current mode.
        // 1. Set P (PID) = 0. Store start P value
        // 2. Set goal poistion to present position
        // 3. Increase current or P (PID)
        // 4. Repeat 2,3 for X seconds.
        Sleep(0.1);
        auto headThread = std::async(std::launch::async, &EpiServos::PowerOn, this, HEAD_ID_MIN, HEAD_ID_MAX, std::ref(portHandlerHead), std::ref(packetHandlerHead));
        auto pupilThread = std::async(std::launch::async, &EpiServos::PowerOnPupil, this); // Different control table.
        auto leftArmThread = std::async(std::launch::async, &EpiServos::PowerOn, this, ARM_ID_MIN, ARM_ID_MAX, std::ref(portHandlerLeftArm), std::ref(packetHandlerLeftArm));
        auto rightArmThread = std::async(std::launch::async, &EpiServos::PowerOn, this, ARM_ID_MIN, ARM_ID_MAX, std::ref(portHandlerRightArm), std::ref(packetHandlerRightArm));
        auto bodyThread = std::async(std::launch::async, &EpiServos::PowerOn, this, BODY_ID_MIN, BODY_ID_MAX, std::ref(portHandlerBody), std::ref(packetHandlerBody));

        if (!headThread.get())
            Notify(msg_fatal_error, "Can not power on head");
        if (!pupilThread.get())
            Notify(msg_fatal_error, "Can not power on pupil");
        if (!leftArmThread.get())
            Notify(msg_fatal_error, "Can not power on left arm");
        if (!rightArmThread.get())
            Notify(msg_fatal_error, "Can not power on right arm");
        if (!bodyThread.get())
            Notify(msg_fatal_error, "Can not power on body");

        std::cout << "Power off robot" << std::endl;
        return true;
    }

    bool PowerOff(int IDMin, int IDMax, dynamixel::PortHandler *portHandler, dynamixel::PacketHandler *packetHandler)
    {
        if (portHandler == NULL) // If no port handler return true. Only return false if communication went wrong.
            return true;

        Timer t;
        const int nrOfServos = IDMax - IDMin + 1;
        int dxl_comm_result = COMM_TX_FAIL; // Communication result
        uint8_t dxl_error = 0;              // Dynamixel error
        uint16_t start_p_value[7] = {0, 0, 0, 0, 0, 0, 0};
        uint32_t present_postition_value[7] = {0, 0, 0, 0, 0, 0, 0};

        // Get P values
        for (int i = 0; i < nrOfServos; i++) {
            if (COMM_SUCCESS != packetHandler->read2ByteTxRx(portHandler, IDMin + i, servoControlTable["Position P Gain"]["Address"], &start_p_value[i], &dxl_error)) {
            Warning("Can not read P value for servo ID: " + std::to_string(IDMin + i));
            return false;
            }
        }

        // t.Reset();
        Warning("Power off servos. If needed, support the robot while power off the servos");

           
        // Turn p to zero
        for (int i = 0; i < nrOfServos; i++) {
            if (COMM_SUCCESS != packetHandler->write2ByteTxRx(portHandler, IDMin + i, servoControlTable["Position P Gain"]["Address"], 0, &dxl_error)) {
            Warning("Can not set P value to 0 for servo ID: " + std::to_string(IDMin + i));    
            return false;
            }
        }

        Sleep(TIMER_POWER_OFF);

        
        // // Set goal position to present postiion
        // for (int i = 0; i < nrOfServos; i++)
        //     if (COMM_SUCCESS != packetHandler->write4ByteTxRx(portHandler, IDMin + i, servoControlTable["Goal Position"]["Address"], present_postition_value[i], &dxl_error))
        //         return false;

        // t.Restart();
        Sleep(TIMER_POWER_OFF_EXTENDED);

        // Enable torque off
        Notify(msg_debug, "Enable torque off");
        for (int i = 0; i < nrOfServos; i++) {
            if (COMM_SUCCESS != packetHandler->write1ByteTxRx(portHandler, IDMin + i, 64, 0, &dxl_error)) {
            Warning("Can not turn off torque for servo ID: " + std::to_string(IDMin + i));
            return false;
            }
        }
        // Set P value to start value
        for (int i = 0; i < nrOfServos; i++) {
            if (COMM_SUCCESS != packetHandler->write2ByteTxRx(portHandler, IDMin + i, 84, start_p_value[i], &dxl_error)) {
            Warning("Can not set P value for servo ID: " + std::to_string(IDMin + i));
            return false;
            }
        }

        return true;
    }

    bool PowerOffPupil()
    {
        uint8_t dxl_error = 0; // Dynamixel error

        // Torque off. No fancy rampiong
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 2, 24, 0, &dxl_error))
            return false;
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 3, 24, 0, &dxl_error))
            return false;

        return true;
    }

    bool PowerOffRobot()
    {

        auto headThread = std::async(std::launch::async, &EpiServos::PowerOff, this, HEAD_ID_MIN, HEAD_ID_MAX, std::ref(portHandlerHead), std::ref(packetHandlerHead));
        auto pupilThread = std::async(std::launch::async, &EpiServos::PowerOffPupil, this);
        auto leftArmThread = std::async(std::launch::async, &EpiServos::PowerOff, this, ARM_ID_MIN, ARM_ID_MAX, std::ref(portHandlerLeftArm), std::ref(packetHandlerLeftArm));
        auto rightArmThread = std::async(std::launch::async, &EpiServos::PowerOff, this, ARM_ID_MIN, ARM_ID_MAX, std::ref(portHandlerRightArm), std::ref(packetHandlerRightArm));
        auto bodyThread = std::async(std::launch::async, &EpiServos::PowerOff, this, BODY_ID_MIN, BODY_ID_MAX, std::ref(portHandlerBody), std::ref(packetHandlerBody));

        if (!headThread.get())
            Notify(msg_fatal_error, "Can not power off head");
        if (!pupilThread.get())
            Notify(msg_fatal_error, "Can not power off pupil");
        if (!leftArmThread.get())
            Notify(msg_fatal_error, "Can not power off left arm");
        if (!rightArmThread.get())
            Notify(msg_fatal_error, "Can not power off right arm");
        if (!bodyThread.get())
            Notify(msg_fatal_error, "Can not power off body");

        // Power down servos.
        // 1. Store P (PID) value
        // 2. Ramp down P
        // 3. Turn of torque enable
        // 4. Set P (PID) valued from 1.

        return (true);
    }
    // XL320 is using 2.0 but with a very limited controltable.
    bool AutoCalibratePupil()
    {
        int dxl_comm_result = COMM_TX_FAIL; // Communication result
        uint8_t dxl_error = 0;              // Dynamixel error
        Timer t;
        double xlTimer = 0.01; // Timer in sec. XL320 need this. Not sure why.

        // Torque off. No fancy rampiong
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 2, 24, 0, &dxl_error))
            return false;
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 3, 24, 0, &dxl_error))
            return false;
        Sleep(xlTimer);

        // Reset min and max limit
        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 2, 6, 0, &dxl_error))
            return false;
        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 3, 6, 0, &dxl_error))
            return false;
        Sleep(xlTimer);

        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 2, 8, 1023, &dxl_error))
            return false;
        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 3, 8, 1023, &dxl_error))
            return false;
        Sleep(xlTimer);

        // Turn down torque limit. To make sure we do not make a fire when hitting the limit.
        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 2, 35, 500, &dxl_error))
            return false;
        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 3, 35, 500, &dxl_error))
            return false;
        Sleep(xlTimer);

        // Torque on. No fancy rampiong
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 2, 24, 1, &dxl_error))
            return false;
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 3, 24, 1, &dxl_error))
            return false;
        Sleep(xlTimer);

        // Go to min pos
        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 2, 30, 0, &dxl_error))
            return false;
        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 3, 30, 0, &dxl_error))
            return false;
        Sleep(1.5); // Sleep for 1.500 ms to get to min position.

        // Read present position
        uint16_t present_postition_value[2] = {0, 0};
        if (COMM_SUCCESS != packetHandlerPupil->read2ByteTxRx(portHandlerPupil, 2, 37, &present_postition_value[0], &dxl_error))
            return false;
        if (COMM_SUCCESS != packetHandlerPupil->read2ByteTxRx(portHandlerPupil, 3, 37, &present_postition_value[1], &dxl_error))
            return false;

        AngleMinLimitPupil[0] = present_postition_value[0] + 5;
        AngleMinLimitPupil[1] = present_postition_value[1] + 5;

        AngleMaxLimitPupil[0] = AngleMinLimitPupil[0] + 80;
        AngleMaxLimitPupil[1] = AngleMinLimitPupil[1] + 80;

        // Not implemented.
        Notify(msg_debug, "Position limits pupil servos (auto calibrate): min " +
                              std::to_string(AngleMinLimitPupil[0]) + " " +
                              std::to_string(AngleMinLimitPupil[1]) + " max " +
                              std::to_string(AngleMaxLimitPupil[0]) + " " +
                              std::to_string(AngleMaxLimitPupil[1]));
        // Torque off. No fancy rampiong
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 2, 24, 0, &dxl_error))
            return false;
        if (COMM_SUCCESS != packetHandlerPupil->write1ByteTxRx(portHandlerPupil, 3, 24, 0, &dxl_error))
            return false;
        Sleep(xlTimer);

        // Set torque limit to max
        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 2, 35, 1023, &dxl_error))
            return false;
        if (COMM_SUCCESS != packetHandlerPupil->write2ByteTxRx(portHandlerPupil, 3, 35, 1023, &dxl_error))
            return false;
        Sleep(xlTimer);

        return true;
    }
    ~EpiServos()
    {
        if (simulate) // no memory to return
        {
            return;
        }

        // Torque down
        PowerOffRobot();

        // Close ports
        if (EpiTorsoMode || EpiFullMode)
        {
            portHandlerHead->closePort();
            portHandlerPupil->closePort();
        }
        if (EpiFullMode)
        {
            portHandlerLeftArm->closePort();
            portHandlerRightArm->closePort();
            portHandlerBody->closePort();
        }

        // Free memory
        delete groupSyncWriteHead;
        delete groupSyncReadHead;
        delete groupSyncWriteLeftArm;
        delete groupSyncReadLeftArm;
        delete groupSyncWriteRightArm;
        delete groupSyncReadRightArm;
        delete groupSyncWriteBody;
        delete groupSyncReadBody;
        // delete groupSyncWritePupil;
    }
};

INSTALL_CLASS(EpiServos)