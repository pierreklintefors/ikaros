sound_name = "disgusted_2" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

# pitch_manual = c(190, 183, 183, 189, 193, 192, 194, 198, 200, 200, 200, 197, 194, 189, 187, 185, 185, 181, 175, 171, 166, 160, 155, 157, 158, 158, 158, 154, 152, 145, 139, 142, 144, 143, 134, 123, 106)

s = soundgen(
  sylLen = 1030,
  ampl = c(0, -10, -10),
  pitch = c(190, 193, 200, 194, 185, 166, 158, 152, 144, 106),
  rolloff = c(-15, -10, -15),
  jitterDep = c(0, 0, .25, 1), shimmerDep = 10,
  formants = '0u',
  formantDep = 1.25,
  mouth = c(.25, 1),
  noise = -30,
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
      sylLen = 600 * intensPars$dur[i],
      ampl = c(0, -10, -10),
      pitch = c(190, 193, 200, 194, 185, 166, 158, 152, 144, 106) * .75 * intensPars$pitch[i] * robotPars$pitch,
      rolloff = c(-15, -10, -15) + intensPars$rolloff[i], 
      jitterDep = c(0, 0, .25, 1) * intensPars$jitter[i], 
      shimmerDep = 10 * intensPars$shimmer[i],
      formants = list(
        f1 = c(640, 300) * robotPars$formants,
        f2 = c(1670, 610) * robotPars$formants,
        f3 = c(2300, 2500) * robotPars$formants,
        f4 = c(3880, 4200) * robotPars$formants
      ),
      formantDep = 1.25,
      mouth = c(.25, 1),
      noise = -30,
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
