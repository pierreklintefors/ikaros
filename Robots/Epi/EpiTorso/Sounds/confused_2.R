sound_name = "confused_2" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

# pitch_manual = c(157, 160, 161, 163, 168, 180, 197, 218, 248, 277, 315, 361, 390, 420, 437, 445, 445, 441, 437, 441) 

s = soundgen(
  sylLen = 510,
  ampl = c(0, -15),
  pitch = c(157, 163, 197, 277, 420, 445, 441),
  jitterDep = .1, shimmerDep = 5,
  rolloff = c(-18, -15, -15, -20),
  formants = '0',
  formantWidth = 1.5,
  mouth = c(.3, .5),
  noise = list(time = c(-20, 510, 600),
               value = c(-20, -25, -35)),
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
      sylLen = 510 * intensPars$dur[i],
      ampl = c(0, -15),
      pitch = c(157, 163, 197, 277, 420, 445, 441) * .5 * intensPars$pitch[i] * robotPars$pitch, 
      rolloff = c(-18, -15, -15, -20) + intensPars$rolloff[i], 
      formants = schwa * robotPars$formants,
      mouth = c(.3, .5),
      noise = list(time = c(-20, 510, 600) * intensPars$dur[i],
                   value = c(-20, -25, -35)),
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
