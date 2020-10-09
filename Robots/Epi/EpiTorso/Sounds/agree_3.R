sound_name = "agree_3" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

# pitch = c(118, 98, 95, 98, 81, 80, 80, 138, 155, 162, 166, 166, 151)

s = soundgen(
  sylLen = 335,
  ampl = c(-12, -12, -25, -3, 0, -5),
  pitch = c(118, 95, 98, 80, 138, 162, 166, 151), 
  rolloff = c(-12, -10, -20, -10, -11, -18),
  formants = mmm,
  formantWidth = 1.5,
  noise = list(time = c(0, 335, 370), value = c(-30, -30, -40)),
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
      sylLen = 335 * intensPars$dur[i],
      ampl = c(-12, -12, -25, -3, 0, -5),
      pitch = c(118, 95, 98, 80, 138, 162, 166, 151) * robotPars$pitch, 
      rolloff = c(-12, -10, -20, -10, -11, -18) + intensPars$rolloff[i],
      formants = mmm * robotPars$formants,
      formantWidth = 1.5,
      noise = list(time = c(0, 335, 370) * intensPars$dur, value = c(-30, -30, -40)),
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
