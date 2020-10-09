sound_name = "disagree_1" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

# pitch_manual = c(118, 126, 130, 122, NA, NA, 113, 119, 119, 118, 112, 112)

s = soundgen(
  nSyl = 2,
  sylLen = c(110, 190),
  pauseLen = 25,
  attackLen = c(15, 80),
  pitch = c(118, 130, 115), pitchGlobal = c(0, -1),
  rolloff = c(-15, -14, -18),
  jitterDep = .3, shimmerDep = 10,
  formants = schwaNas,
  noise = list(time = c(0, 110, 150),
               value = c(-25, -25, -30)),
  rolloffNoise = -6,
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
      nSyl = 2,
      sylLen = c(110, 190) * intensPars$dur[i],
      pauseLen = 35 * intensPars$dur[i],
      attackLen = c(15, 80),
      pitch = c(118, 130, 115) * .75 * intensPars$pitch[i] * robotPars$pitch,
      pitchGlobal = c(0, -1),
      rolloff = c(-15, -14, -18) + intensPars$rolloff[i], 
      jitterDep = .1 * intensPars$jitter[i], 
      shimmerDep = 2 * intensPars$shimmer[i],
      formants = list(
        f1 = list(freq = 750 * robotPars$formants, width = 120, amp = 10),
        f1.7 = list(freq = 1400 * robotPars$formants, width = 80, amp = -10),
        f1.8 = list(freq = 1570 * robotPars$formants, width = 100, amp = 15),
        f2 = list(freq = 1900 * robotPars$formants, width = 150, amp = 25),
        f2.7 = list(freq = 2300 * robotPars$formants, width = 150, amp = -15),
        f3 = list(freq = 2670 * robotPars$formants, width = 80, amp = 30),
        f3.5 = list(freq = 3500 * robotPars$formants, width = 100, amp = -15),
        f3.7 = list(freq = 3780 * robotPars$formants, width = 100, amp = 15),
        f4 = list(freq = 4340 * robotPars$formants, amp = 25),
        f4.5 = list(freq = 4900 * robotPars$formants, width = 120, amp = -10),
        f5 = list(freq = 5300 * robotPars$formants, width = 120, amp = 10)
      ),
      noise = list(time = c(0, 110, 150) * intensPars$dur[i],
                   value = c(-25, -25, -30)),
      rolloffNoise = -6,
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
