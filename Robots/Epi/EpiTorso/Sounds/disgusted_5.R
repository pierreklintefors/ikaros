sound_name = "disgusted_5" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

# pitch_manual = c(181, 189, 188, 188, 185, 181, 181)

s = soundgen(
  sylLen = 240,
  ampl = c(-2, 0, -10),
  pitch = c(181, 189, 185, 181),
  rolloff = c(-11, -7),
  jitterDep = .25, shimmerDep = 10,
  formants = '0a',
  mouth = c(.3, .55), 
  noise = list(time = c(0, 200, 240, 400),
               value = c(-30, -10, -5, -35)),
  rolloffNoise = c(-6, -2),
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
      sylLen = 240 * intensPars$dur[i],
      ampl = c(-2, 0, -10),
      pitch = c(181, 189, 185, 181) * .75 * intensPars$pitch[i] * robotPars$pitch,
      rolloff = c(-11, -7) + intensPars$rolloff[i], 
      jitterDep = .25 * intensPars$jitter[i], 
      shimmerDep = 10 * intensPars$shimmer[i],
      formants = list(
        f1 = c(440, 860) * robotPars$formants,
        f2 = c(1270, 1430) * robotPars$formants,
        f3 = c(2300, 2900) * robotPars$formants,
        f4 = c(3880, 4200) * robotPars$formants
      ),
      mouth = c(.6, .3), 
      noise = list(time = c(0, 200, 240, 400) * intensPars$dur[i],
                   value = c(-30, -10, -5, -35)),
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
