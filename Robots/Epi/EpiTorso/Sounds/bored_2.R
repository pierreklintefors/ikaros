sound_name = "bored_2" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

s = soundgen(
  sylLen = 470,
  ampl = c(0, -15),
  pitch = c(104, 117, 107, 93, 95, 78), 
  jitterDep = c(.7, 1.3, .5), shimmerDep = c(0, 20, 5),
  rolloff = c(-19, -16, -25, -26, -27),
  formants = '0',
  mouth = c(.55, .45, .4),
  noise = list(time = c(0, 150, 470, 550), 
               value = c(-25, -20, -30, -40)),
  rolloffNoise = 0, rolloffNoiseExp = -2,
  temperature = .05, 
  addSilence = 50,
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
      sylLen = 470 * intensPars$dur[i],
      ampl = c(0, -15),
      pitch = c(104, 117, 107, 93, 95, 78) * .5 * (1 / intensPars$pitch[i]) * robotPars$pitch,   # inverse rel with f0: the more bored, the lower
      jitterDep = c(.7, 1.3, .5) / 2, shimmerDep = c(0, 20, 5) / 2,
      rolloff = c(-19, -16, -25, -26, -27) + intensPars$rolloff[i], 
      formants = schwa * robotPars$formants,
      mouth = c(.55, .45, .4),
      noise = list(time = c(0, 150, 470, 550) * intensPars$dur[i], 
                   value = c(-25, -20, -30, -40)),
      rolloffNoise = 0, rolloffNoiseExp = -2,
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
