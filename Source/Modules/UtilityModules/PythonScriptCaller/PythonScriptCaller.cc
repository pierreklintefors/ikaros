#include "ikaros.h"
#include <string>
#include <vector>
#include <sstream> // For parsing input string if needed, not directly used with matrix
#include <fstream> // General C++ utility
#include <memory>  // For unique_ptr or shared_ptr if used
#include <chrono>  // For sleep and timeouts
#include <thread>  // For std::this_thread::sleep_for

// Shared memory specific includes
#include <sys/mman.h>
#include <sys/stat.h> // For mode constants
#include <fcntl.h>    // For O_* constants
#include <unistd.h>   // For fork, execl, sleep, close, ftruncate, shm_unlink
#include <sys/wait.h> // For waitpid
#include <cstring>    // For strerror, memset, memcpy
#include <atomic>     // For std::atomic_thread_fence
#include <iomanip>    // For std::fixed, std::setprecision (if formatting output string)

using namespace ikaros;

// Structure for the control flags in shared memory
struct PythonScriptCallerSharedFlags {
    bool cpp_wrote_input;   // Set by C++ when new input is ready for Python
    bool python_wrote_output; // Set by Python when new output is ready for C++
    bool shutdown_signal;   // Set by C++ to signal Python to terminate
    // Padding to ensure proper alignment if followed by other data types,
    // or just to make the struct size a multiple of a common alignment (e.g., 4 or 8 bytes).
    // If only bools, and data follows contiguously, this might be less critical
    // but good practice if data alignment matters for performance or direct casting.
    // For now, let's calculate padding to make it a multiple of 4 bytes (common float alignment)
    // assuming bool is 1 byte. (3 bools = 3 bytes). To reach 4 bytes, add 1 byte padding.
    // If sizeof(bool) is different on some systems, this calculation might need to be more robust.
    // A simpler approach if data follows directly is to ensure the data pointer is aligned.
    // Given data_ptr is float*, and flags_size is sizeof(this_struct), mmap typically aligns.
    // Let's use the original padding logic for now, but based on only 3 bools.
    char padding[(sizeof(float) - ((sizeof(bool) * 3) % sizeof(float))) % sizeof(float)];
    // REMOVED: parameter num_inputs_param;
    // REMOVED: parameter num_outputs_param;
    // REMOVED: parameter shared_memory_name_param;
};


class PythonScriptCaller: public Module {


    // Parameters from .ikc
    parameter script_path_param;
    parameter venv_path_param;
    parameter num_inputs_param;
    parameter num_outputs_param;
    parameter shared_memory_name_param;
    parameter timeout_param;
    // Inputs & Outputs from .ikc (assuming type="matrix")
    matrix input_matrix;
    matrix output_matrix;

    // Shared memory variables
    int shm_fd = -1;
    void* shm_ptr = MAP_FAILED;
    size_t shm_total_size = 0;
    PythonScriptCallerSharedFlags* shm_flags_ptr = nullptr;
    float* shm_data_ptr = nullptr;

    std::string actual_shm_name;
    int num_inputs_val = 0;
    int timeout_ms;
    // Process management
    pid_t python_process_pid = -1;


    void Init() {

        Bind(script_path_param, "ScriptPath");
        Bind(venv_path_param, "VenvPath");
        Bind(num_inputs_param, "NumberInputs");
        Bind(num_outputs_param, "NumberOutputs");
        Bind(shared_memory_name_param, "SharedMemoryName");
        Bind(timeout_param, "Timeout");
        Bind(input_matrix, "Input");
        Bind(output_matrix, "Output");

        std::string script_path = script_path_param.as_string();
        std::string venv_path = venv_path_param.as_string();
        num_inputs_val = num_inputs_param.as_int();
        int num_outputs_val = num_outputs_param.as_int();
        std::string shm_base_name = shared_memory_name_param.as_string();
        timeout_ms = timeout_param.as_int();
        
        if (script_path.empty()) { Error("Parameter 'ScriptPath' must be set."); return; }
        if (venv_path.empty()) { Error("Parameter 'VenvPath' must be set."); return; }
        if (shm_base_name.empty()) { Error("Parameter 'SharedMemoryName' must be set."); return; }
        if (num_inputs_val <= 0) { Error("Parameter 'NumberInputs' must be a positive integer."); return; }
        if (num_outputs_val <= 0) { Error("Parameter 'NumberOutputs' must be a positive integer."); return; }

        actual_shm_name = "/" + shm_base_name;

        size_t flags_size = sizeof(PythonScriptCallerSharedFlags);
        size_t data_array_num_floats = num_inputs_val + 1 + num_outputs_val;
        size_t data_size = data_array_num_floats * sizeof(float);
        shm_total_size = flags_size + data_size;

        shm_unlink(actual_shm_name.c_str());

        shm_fd = shm_open(actual_shm_name.c_str(), O_CREAT | O_RDWR, 0666);
        if (shm_fd == -1) {
            Error("Init: shm_open failed for " + actual_shm_name + ": " + strerror(errno));
            return;
        }

        if (ftruncate(shm_fd, shm_total_size) == -1) {
            Error("Init: ftruncate failed for " + actual_shm_name + ": " + strerror(errno));
            close(shm_fd); shm_unlink(actual_shm_name.c_str()); shm_fd = -1;
            return;
        }

        shm_ptr = mmap(NULL, shm_total_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (shm_ptr == MAP_FAILED) {
            Error("Init: mmap failed for " + actual_shm_name + ": " + strerror(errno));
            close(shm_fd); shm_unlink(actual_shm_name.c_str()); shm_fd = -1; shm_ptr = nullptr;
            return;
        }

        shm_flags_ptr = static_cast<PythonScriptCallerSharedFlags*>(shm_ptr);
        shm_data_ptr = reinterpret_cast<float*>(static_cast<char*>(shm_ptr) + flags_size);

        memset(shm_ptr, 0, shm_total_size);
        shm_flags_ptr->cpp_wrote_input = false;
        shm_flags_ptr->python_wrote_output = false;
        shm_flags_ptr->shutdown_signal = false;

        Debug("Shared memory '" + actual_shm_name + "' initialized. Total size: " + std::to_string(shm_total_size) +
            " bytes. Expected inputs: " + std::to_string(num_inputs_val) +
            ", Max Python outputs: " + std::to_string(num_outputs_val));

        StartPythonProcess();
        if (python_process_pid > 0) {
            Sleep(0.2); // Give Python a moment to start
        }
    }

    void StartPythonProcess() {
        std::string venv_python_path = venv_path_param.as_string();
        std::string script_full_path = script_path_param.as_string();

        python_process_pid = fork();

        if (python_process_pid < 0) {
            Error("StartPythonProcess: fork failed: " + std::string(strerror(errno)));
            return;
        }

        if (python_process_pid == 0) { // Child process
            std::string shm_name_for_script = shared_memory_name_param.as_string(); // Pass base name
            std::string num_inputs_str = std::to_string(num_inputs_val);
            std::string num_outputs_str = std::to_string(num_outputs_param.as_int());
            std::string flags_size_str = std::to_string(sizeof(PythonScriptCallerSharedFlags));


            execl(venv_python_path.c_str(), venv_python_path.c_str(), script_full_path.c_str(),
                shm_name_for_script.c_str(), num_inputs_str.c_str(), num_outputs_str.c_str(),
                flags_size_str.c_str(), (char*)nullptr);

            // If execl returns, an error occurred
            // Using _exit in child to avoid calling destructors of Ikaros objects
            fprintf(stderr, "[PythonScriptCaller CHILD ERROR] execl failed for %s %s: %s\n",
                    venv_python_path.c_str(), script_full_path.c_str(), strerror(errno));
            _exit(EXIT_FAILURE);
        }
        Debug("Python script process started with PID: " + std::to_string(python_process_pid));
    }

    void Tick() {
        try {
            Debug("PythonScriptCaller::Tick() called. Current Tick: " + std::to_string(GetTick()));
            // Moving input_matrix.info() to the very top to check its state immediately.
            // If input_matrix itself is problematic, info() might throw.
            // input_matrix.info(); // Uncomment this if the GetTick() line above works without error.
        } catch (const std::exception& e) {
            Error("Tick: Exception during initial GetTick() debug: " + std::string(e.what()));
            return;
        } catch (...) {
            Error("Tick: Unknown exception during initial GetTick() debug.");
            return;
        }

        try {
            if (!input_matrix.connected()) { // Potential source of error
                Error("Tick: Input matrix is not connected for PythonScriptCaller.");
                return;
            }
        } catch (const std::exception& e) {
            Error("Tick: Exception during input_matrix.connected() check: " + std::string(e.what()) +
                  ". input_matrix.size() (if accessible): " + (input_matrix.connected() ? std::to_string(input_matrix.size()) : "N/A"));
            return;
        } catch (...) {
            Error("Tick: Unknown exception during input_matrix.connected() check.");
            return;
        }

        try {
            if (shm_ptr == MAP_FAILED || shm_flags_ptr == nullptr || shm_data_ptr == nullptr || python_process_pid <= 0) {
                Error("Tick: Shared memory or Python process not properly initialized.");
                return;
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            if (shm_flags_ptr->shutdown_signal) { // Potential source if shm_flags_ptr is bad
                Warning("Tick: Shutdown signal is active. Python process might be terminating.");
                return;
            }
        } catch (const std::exception& e) {
            Error("Tick: Exception during SHM/process/shutdown_signal check: " + std::string(e.what()));
            return;
        } catch (...) {
            Error("Tick: Unknown exception during SHM/process/shutdown_signal check.");
            return;
        }

        // The previous input_matrix.info() call by the user was around here.
        // The GetTick() > 1 check also follows here.

        if (GetTick() > 1) {
            // Check if Python is ready for new input OR if C++ hasn't written anything yet (initial state).
            std::atomic_thread_fence(std::memory_order_acquire);
            if (shm_flags_ptr->python_wrote_output || !shm_flags_ptr->cpp_wrote_input) {
                if (input_matrix.size() != num_inputs_val) {
                    Error("Tick: Connected input_matrix size (" + std::to_string(input_matrix.size()) +
                        ") does not match \'NumberInputs\' parameter (" + std::to_string(num_inputs_val) + ").");
                    return;
                }

                for (int i = 0; i < num_inputs_val; ++i) {
                    try {
                        shm_data_ptr[i] = input_matrix(i);
                    } catch (const std::exception& e) {
                        Error("Tick: Exception while accessing input_matrix(" + std::to_string(i) + "). Error: " + e.what() +
                              ". input_matrix.size()=" + std::to_string(input_matrix.size()) +
                              ", num_inputs_val=" + std::to_string(num_inputs_val));
                        // Optionally, re-throw or handle more gracefully if needed, for now just error out.
                        return; // Stop processing this Tick
                    } catch (...) {
                        Error("Tick: Unknown exception while accessing input_matrix(" + std::to_string(i) + ")." +
                              ". input_matrix.size()=" + std::to_string(input_matrix.size()) +
                              ", num_inputs_val=" + std::to_string(num_inputs_val));
                        return; // Stop processing this Tick
                    }
                }
                // Debug("Tick: Wrote " + std::to_string(num_inputs_val) + " inputs to SHM.");

                shm_flags_ptr->python_wrote_output = false;
                std::atomic_thread_fence(std::memory_order_release);
                shm_flags_ptr->cpp_wrote_input = true;
                std::atomic_thread_fence(std::memory_order_release);
            }

            auto start_time = std::chrono::steady_clock::now();
            bool got_response = false;

            while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() < timeout_ms)
            {
                std::atomic_thread_fence(std::memory_order_acquire);
                if (shm_flags_ptr->shutdown_signal)
                {
                    Warning("Tick: Shutdown signal detected while waiting for Python.");
                    return;
                }
                if (shm_flags_ptr->python_wrote_output)
                {
                    got_response = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }

            if (got_response)
            {
                std::atomic_thread_fence(std::memory_order_acquire); // Ensure visibility of data written by Python
                float *output_section_ptr = shm_data_ptr + num_inputs_val;
                int actual_output_count = static_cast<int>(output_section_ptr[0]);

                if (actual_output_count < 0 || actual_output_count > num_outputs_param.as_int())
                {
                    Error("Tick: Python script reported invalid number of outputs: " + std::to_string(actual_output_count) +
                          ". Max expected: " + std::to_string(num_outputs_param.as_int()));
                    output_matrix.resize(0); // Indicate error
                    // Reset flags defensively
                    shm_flags_ptr->cpp_wrote_input = false; // Allow C++ to try writing again
                    std::atomic_thread_fence(std::memory_order_release);
                    // python_wrote_output is already true, will be set to false by C++ on next write cycle.
                    return;
                }

                output_matrix.resize(actual_output_count);
                if (actual_output_count > 0)
                {
                    memcpy(output_matrix.data(), &output_section_ptr[1], actual_output_count * sizeof(float));
                }
              

                // The flags are now: cpp_wrote_input=false (set by Python), python_wrote_output=true (set by Python)
                // This state is correct for the next C++ write cycle.
            }
            else
            {
                Warning("Tick: Timeout waiting for Python script response. Tick: " + std::to_string(GetTick()));
                // output_matrix remains unchanged or could be cleared.
            }
        }
        
        
    }

    ~PythonScriptCaller() {
        StopPythonProcess();

        if (shm_ptr != MAP_FAILED && shm_ptr != nullptr) {
            if (munmap(shm_ptr, shm_total_size) == -1) {
                // Error("Destructor: munmap failed for " + actual_shm_name + ": " + strerror(errno));
            }
            shm_ptr = MAP_FAILED;
        }

        if (shm_fd != -1) {
            close(shm_fd); // Error check omitted for brevity in destructor path
            shm_fd = -1;
        }

        if (!actual_shm_name.empty()) {
            shm_unlink(actual_shm_name.c_str()); // Error check omitted
        }
        Debug("PythonScriptCaller destroyed. Shared memory for '" + actual_shm_name + "' cleaned up.");
    }

    void StopPythonProcess() {
        if (python_process_pid <= 0) return;

        Debug("Stopping Python process PID: " + std::to_string(python_process_pid));

        if (shm_flags_ptr != nullptr && shm_ptr != MAP_FAILED) {
            std::atomic_thread_fence(std::memory_order_acquire); // Ensure we don't overwrite Python's operation
            shm_flags_ptr->shutdown_signal = true;
            std::atomic_thread_fence(std::memory_order_release);
            // Debug("Shutdown signal sent to Python process via SHM.");
        }

        int stat;
        pid_t result_pid;
        auto shutdown_timeout_start = std::chrono::steady_clock::now();
        const int grace_period_ms = 500; // 0.5 seconds

        while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - shutdown_timeout_start).count() < grace_period_ms) {
            result_pid = waitpid(python_process_pid, &stat, WNOHANG);
            if (result_pid == python_process_pid) {
                Debug("Python process " + std::to_string(python_process_pid) + " terminated gracefully.");
                python_process_pid = -1; return;
            }
            if (result_pid == -1 && errno != ECHILD) { /* Error("StopPythonProcess: waitpid error: " + std::string(strerror(errno))); */ break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        // Warning("Python process " + std::to_string(python_process_pid) + " did not terminate via SHM signal or timed out. Sending SIGTERM.");
        if (kill(python_process_pid, SIGTERM) == 0) { // If kill returns 0, signal was sent
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Brief wait after SIGTERM
            result_pid = waitpid(python_process_pid, &stat, WNOHANG);
            if (result_pid == python_process_pid) {
                Debug("Python process " + std::to_string(python_process_pid) + " terminated after SIGTERM.");
                python_process_pid = -1; return;
            }
        }

        // Warning("Python process " + std::to_string(python_process_pid) + " did not terminate after SIGTERM. Sending SIGKILL.");
        kill(python_process_pid, SIGKILL); // Error check omitted for brevity
        waitpid(python_process_pid, &stat, 0); // Blocking wait to reap zombie
        Debug("Python process " + std::to_string(python_process_pid) + " terminated after SIGKILL.");
        python_process_pid = -1;
    }
};

INSTALL_CLASS(PythonScriptCaller);
