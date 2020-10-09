sound_name = "pleased_4"

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

# pitch_manual = c(108, 134, 143, 158, 163, 170, 171, 167, 122, 81, 63, 77, 63, 54, 59, 59, 59, 59, 59, 59, 59)

s = soundgen(
  sylLen = 630,
  attackLen = 150,
  # pitch = c(108, 158, 167, 126, 108, 100, 100),
  pitch = c(108, 158, 167, 63, 54, 59, 59),
  rolloff = c(-20, -20, -10, -15, -25),
  glottis = c(15, 15, 0, 0, 10, 200),
  # amDep = c(35, 25, 0, 0, 70, 100), amFreq = 25, amShape = .7,
  formants = mmm,
  # formantWidth = 2, formantDep = 1.5,
  noise = -35,
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
      sylLen = 630 * intensPars$dur[i],
      ampl = c(0, -10),
      attackLen = 150,
      pitch = c(108, 158, 167, 63, 54, 59, 59) * intensPars$pitch[i] * robotPars$pitch,
      rolloff = c(-20, -20, -10, -15, -25) + intensPars$rolloff[i],
      # glottis = c(15, 15, 0, 10, 200, 400), 
      formants = mmm * robotPars$formants,
      noise = -35,
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
