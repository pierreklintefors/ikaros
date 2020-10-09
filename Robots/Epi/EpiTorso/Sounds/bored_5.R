sound_name = "bored_5" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

s = soundgen(
  pitch = NULL,
  formants = '0',
  formantWidth = 1.5,
  mouth = 0,
  noise = list(time = c(0, 160, 840), 
               value = c(-20, 0, -40)),
  rolloffNoise = 7, rolloffNoiseExp = -2,
  temperature = .05, 
  addSilence = c(50, 150),
  samplingRate = 22050, pitchSamplingRate = 22050, play = TRUE,
  plot = TRUE, ylim = c(0, 7)
)

seewave::savewav(s, f = 22050, filename = paste0(sound_name, '_orig.wav'))

##############
# R O B O T   V O I C E
##############
# NB: instability caused by nasalization with F1 around 550 Hz; to avoid this,
# we remove the closed mouth and specify formants manually instead
for (n in 1:nIter) {
  for (i in 1:nrow(intensPars)) {
    temp = soundgen(
      pitch = NULL,
      formants = mmm * robotPars$formants,
      noise = list(time = c(0, 160, 840) * intensPars$dur[i], 
                   value = c(-20, 0, -40)),
      rolloffNoise = 0,
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
