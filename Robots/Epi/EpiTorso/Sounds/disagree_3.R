sound_name = "disagree_3" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

# pitch_manual = c(187, 181, 186, 184, 188, 186, 181, 156, 145, 133, 129, 127, 132, 140, 139, 143, 140, 138)

s1 = soundgen(
  sylLen = 100,
  attackLen = c(15, 80),
  pitch = c(170, 190),
  rolloff = -12,
  formants = mmm,
  formantWidth = 1.5,
  noise = list(time = c(0, 100, 150),
               value = c(-25, -25, -30)),
  rolloffNoise = -4,
  temperature = .05, 
  addSilence = 50,
  samplingRate = 22050, pitchSamplingRate = 22050, play = TRUE,
  plot = TRUE, ylim = c(0, 7)
)

s2 = soundgen(
  sylLen = 140,
  attackLen = c(15, 80),
  pitch = c(180, 140),
  rolloff = -12,
  formants = mmm,
  formantWidth = 1.5,
  noise = list(time = c(0, 100, 200),
               value = c(-25, -25, -30)),
  rolloffNoise = -4,
  temperature = .05, 
  addSilence = c(50, 150),
  samplingRate = 22050, pitchSamplingRate = 22050, play = TRUE,
  plot = TRUE, ylim = c(0, 7)
)

s = addVectors(s1, s2, insertionPoint = 22050 * .17)
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
      pitch = c(170, 190) * .75 * intensPars$pitch[i] * robotPars$pitch,
      rolloff = -12 + intensPars$rolloff[i],
      formants = mmm * robotPars$formants,
      formantWidth = 1.5,
      noise = list(time = c(0, 100, 150) * intensPars$dur[i],
                   value = c(-25, -25, -30)),
      rolloffNoise = -4,
      invalidArgAction = 'ignore',
      temperature = temperature, 
      tempEffects = tempEffects,
      addSilence = 50,
      samplingRate = 22050, pitchSamplingRate = 22050, play = TRUE,
      plot = FALSE, ylim = c(0, 7)
    )
    
    temp2 = soundgen(
      sylLen = 140 * intensPars$dur[i],
      attackLen = c(15, 80),
      pitch = c(180, 140) * .75 * intensPars$pitch[i] * robotPars$pitch,
      rolloff = -12 + intensPars$rolloff[i],
      formants = mmm * robotPars$formants,
      formantWidth = 1.5,
      noise = list(time = c(0, 100, 200) * intensPars$dur[i],
                   value = c(-25, -25, -30)),
      rolloffNoise = -4,
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

