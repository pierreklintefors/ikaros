sound_name = "confused_1" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

# pitch_manual = c(107, 105, 102, 101, 102, 110, 134, 153, 182, 205, 217, 224, 212)

s = soundgen(
  sylLen = 345,
  ampl = c(0, -15),
  pitch = c(107, 102, 102, 134, 182, 217, 212, 210),
  rolloff = c(-16, -18),
  formants = '0',
  formantWidth = 1.5,
  noise = list(time = c(-90, 100, 340, 420), 
               value = c(-30, -25, -30, -35)),
  rolloffNoise = -4,
  temperature = .05, 
  addSilence = c(50, 150),
  samplingRate = 22050, pitchSamplingRate = 22050, play = TRUE,
  plot = TRUE, ylim = c(0, 7)
)

seewave::savewav(s, f = 22050, filename = paste0(sound_name, '_orig.wav'))

##############
# R O B O T   V O I C E
##############
for (n in 1:nIter) {
  for (i in 1:nrow(intensPars)) {
    temp = soundgen(
      sylLen = 345 * intensPars$dur[i],
      ampl = c(0, -15),
      pitch = c(107, 102, 102, 134, 182, 217, 212, 210) * intensPars$pitch[i] * robotPars$pitch,
      rolloff = c(-16, -18) + intensPars$rolloff[i], 
      formants = schwa * robotPars$formants,
      formantWidth = 1.5,
      noise = list(time = c(-90, 100, 340, 420) * intensPars$dur[i], 
                   value = c(-30, -25, -30, -35)),
      rolloffNoise = -4,
      invalidArgAction = 'ignore',
      temperature = temperature, 
      tempEffects = tempEffects,
      addSilence = c(50, 150),
      samplingRate = 22050, pitchSamplingRate = 22050, play = TRUE,
      plot = FALSE, ylim = c(0, 7)
    )
    temp_wav = tuneR::Wave(left = temp, samp.rate = 22050, bit = 16)
    temp_wav_norm = tuneR::normalize(temp_wav, level = intensPars$level[i])
    filename = paste0(sound_name, '_', intensPars$intensity[i], '_', n, '.wav')
    seewave::savewav(temp_wav_norm, f = 22050, filename = filename)
  }
}
