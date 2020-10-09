sound_name = "pleased_3" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

# pitch_manual = c(199, 228, 297, 403, 486, 519, 525, 525, 519, 513, 479, 464, 445, 428, 412, 397, 380, 364, 350, 336, 324, 317, 306, 293, 278, 272, 249, 237, 223, 208, 190, 173, 149, 153, 159)

s = soundgen(
  sylLen = 900,
  ampl = c(0, -10),
  attackLen = 150,
  pitch = c(199, 525, 464, 364, 293, 223, 159),
  rolloff = c(-10, -9, -10, -12) + 3, rolloffOct = -.5,
  jitterDep = .05, shimmerDep = 2,
  formants = mmm,
  formantWidth = 1.7,
  mouth = c(.5, .6),
  noise = -40,
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
      sylLen = 900 / 1.25 * intensPars$dur[i],
      ampl = c(0, -10),
      attackLen = 150,
      pitch = c(200, 525, 464, 364, 293, 223, 159) * .5 * intensPars$pitch[i] * robotPars$pitch,
      rolloff = c(-15, -9, -10, -12) + 0 + intensPars$rolloff[i], 
      rolloffOct = -.5,
      jitterDep = .1 * intensPars$jitter[i], 
      shimmerDep = 2 * intensPars$shimmer[i],
      formants = mmm * robotPars$formants,
      formantWidth = 1.7,
      mouth = c(.5, .6),
      noise = -40,
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
