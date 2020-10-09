sound_name = "disagree_5" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

# pitch_manual = c(NA, 121, 125, 132, NA, 96, 86, 97, 98, 99, 97)  # 2 syllables

s1 = soundgen(
  sylLen = 130,
  attackLen = c(15, 80),
  pitch = c(120, 130),
  rolloff = -8,
  formants = mmm,
  formantWidth = 1.5,
  noise = -35,
  temperature = .05, 
  addSilence = c(50, 150),
  samplingRate = 22050, pitchSamplingRate = 22050, play = TRUE,
  plot = TRUE, ylim = c(0, 7)
)

s2 = soundgen(
  sylLen = 190,
  attackLen = c(15, 80),
  pitch = c(95, 90, 100),
  rolloff = c(-8, -8, -8, -15),
  formants = mmm,
  formantWidth = 1.5,
  noise = -35,
  temperature = .05, 
  addSilence = 50,
  samplingRate = 22050, pitchSamplingRate = 22050, play = TRUE,
  plot = TRUE, ylim = c(0, 7)
)

s = addVectors(s1, s2, insertionPoint = 22050 * .21)
playme(s, 22050)

seewave::savewav(s, f = 22050, filename = paste0(sound_name, '_orig.wav'))

##############
# R O B O T   V O I C E
##############
for (n in 1:nIter) {
  for (i in 1:nrow(intensPars)) {
    temp1 = soundgen(
      sylLen = 130 * intensPars$dur[i],
      attackLen = c(15, 80),
      pitch = c(120, 130) * intensPars$pitch[i] * robotPars$pitch,
      rolloff = -8 + intensPars$rolloff[i],
      formants = mmm * robotPars$formants,
      formantWidth = 1.5,
      noise = -35,
      invalidArgAction = 'ignore',
      temperature = temperature, 
      tempEffects = tempEffects,
      addSilence = 50,
      samplingRate = 22050, pitchSamplingRate = 22050, play = TRUE,
      plot = FALSE, ylim = c(0, 7)
    )
    
    temp2 = soundgen(
      sylLen = 190 * intensPars$dur[i],
      attackLen = c(15, 80),
      pitch = c(95, 90, 100) * intensPars$pitch[i] * robotPars$pitch,
      rolloff = c(-8, -8, -8, -15) + intensPars$rolloff[i],
      formants = mmm * robotPars$formants,
      formantWidth = 1.5,
      noise = -35,
      invalidArgAction = 'ignore',
      temperature = temperature, 
      tempEffects = tempEffects,
      addSilence = c(50, 150),
      samplingRate = 22050, pitchSamplingRate = 22050, play = TRUE,
      plot = FALSE, ylim = c(0, 7)
    )
    
    temp = addVectors(temp1, temp2, insertionPoint = length(temp1) + 22050 * .06)
    temp_wav = tuneR::Wave(left = temp, samp.rate = 22050, bit = 16)
    temp_wav_norm = tuneR::normalize(temp_wav, level = intensPars$level[i])
    filename = paste0(sound_name, '_', intensPars$intensity[i], '_', n, '.wav')
    seewave::savewav(temp_wav_norm, f = 22050, filename = filename)
  }
}
