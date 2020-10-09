sound_name = "disagree_4" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

# pitch_manual = c(223, 230, 231, 230, 220, 210, 205, 200, 203, 200, 194, 192, 193, 195, 188, 185)

s1 = soundgen(
  sylLen = 100,
  attackLen = c(15, 80),
  pitch = c(220, 230),
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
  sylLen = 230,
  attackLen = c(15, 80),
  pitch = c(200, 185),
  rolloff = c(-8, -15),
  formants = mmm,
  formantWidth = 1.5,
  noise = -35,
  temperature = .05, 
  addSilence = c(50, 150),
  samplingRate = 22050, pitchSamplingRate = 22050, play = TRUE,
  plot = TRUE, ylim = c(0, 7)
)

s = addVectors(s1, s2, insertionPoint = 22050 * .18)
playme(s, 22050)

seewave::savewav(s, f = 22050, filename = paste0(sound_name, '_orig.wav'))

##############
# R O B O T   V O I C E
##############
for (n in 1:nIter) {
  for (i in 1:nrow(intensPars)) {
    temp1 = soundgen(
      sylLen = 100 * intensPars$dur[i],
      attackLen = c(15, 80),
      pitch = c(220, 230) * .5 * intensPars$pitch[i] * robotPars$pitch,
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
      sylLen = 230 * intensPars$dur[i],
      attackLen = c(15, 80),
      pitch = c(200, 185) * .5 * intensPars$pitch[i] * robotPars$pitch,
      rolloff = c(-8, -15) + intensPars$rolloff[i],
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
    
    temp = addVectors(temp1, temp2, insertionPoint = 22050 * .17)
    temp_wav = tuneR::Wave(left = temp, samp.rate = 22050, bit = 16)
    temp_wav_norm = tuneR::normalize(temp_wav, level = intensPars$level[i])
    filename = paste0(sound_name, '_', intensPars$intensity[i], '_', n, '.wav')
    seewave::savewav(temp_wav_norm, f = 22050, filename = filename)
  }
}
