sound_name = "bored_1" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

# pitch = c(194, 202, 209, 215, 217, 219, 217, 216, 217, 211, 207, 203, 200, 198, 195, 194, 194, 194, 194, 195, 195, 195, 193, 193, 192, 189, 185, 183, 181, 175, 174, 170, 172, 167, 168, 168, 164, 163, 159, 157, 157, 160, 161, 154, 147, 144, 150, 144, 154, 152, 149, 148, 149, 153, 149, 137, 146, 149)
# formantsNoise = presets$M1$Formants$consonants$h
# formantsNoise = formantsNoise[3:length(formantsNoise)]

s = soundgen(
  sylLen = 1720,
  ampl = c(0, -2, 0, -10),
  pitch = c(194, 217, 198, 195, 189, 172, 159, 147, 148, 149), 
  rolloff = c(-16, -18, -25) + intensPars$rolloff[i],
  formants = '00a',
  mouth = c(.6, .55, .4),
  noise = list(time = c(0, 600, 1100, 1720, 2000), 
               value = c(-25, -25, -20, -15, -30)),
  # formantsNoise = formantsNoise,
  rolloffNoise = 0, rolloffNoiseExp = -4,
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
      sylLen = 1720 * intensPars$dur[i],
      ampl = c(0, -2, 0, -10),
      pitch = c(194, 217, 198, 195, 189, 172, 159, 147, 148, 149) * .5 * (1 / intensPars$pitch[i]) * robotPars$pitch,   # inverse rel with f0: the more bored, the lower
      rolloff = c(-16, -18, -25) + intensPars$rolloff[i], 
      formants = list(f1 = c(640, 640, 860) * robotPars$formants,
                      f2 = c(1670, 1670, 1430) * robotPars$formants,
                      f3 = c(2700, 2700, 2900) * robotPars$formants),
      mouth = c(.6, .55, .4),
      noise = list(time = c(0, 600, 1100, 1720, 2000) * intensPars$dur[i], 
                   value = c(-25, -25, -20, -15, -30)),
      rolloffNoise = 0, rolloffNoiseExp = -4,
      invalidArgAction = 'ignore',
      temperature = temperature / 2, 
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
