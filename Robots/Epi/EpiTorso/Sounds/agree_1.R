sound_name = "agree_1" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

# pitch = c(115, 115, 115, 112, 110, 107, 109, 125, 137, 149, 159, 169, 178, 185, 188, 195)

s = soundgen(
  sylLen = 380,            # duration of voiced fragment
  attackLen = c(50, 70),   # fade-in/out (sharp or gradual attack)
  pitch = c(115, 112, 109, 149, 195, 165),   # intonation
  rolloff = c(-12, -10, -12, -25, -12, -10, -12, -18),  # breathy-creaky voice
  formants = mmm,                                       # vowel quality
  formantWidth = 1.5,                                   # vowel quality
  noise = -40,              # unvoiced component (breathing)
  temperature = .05,        # how much parameters vary randomly
  addSilence = c(50, 150),  # silence before/after the synthesized sound
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
      sylLen = 380 * intensPars$dur[i],
      attackLen = c(50, 70),
      pitch = c(115, 112, 109, 149, 195, 165) * robotPars$pitch, 
      rolloff = c(-12, -10, -12, -25, -12, -10, -12, -18) + intensPars$rolloff[i],
      formants = mmm * robotPars$formants,
      formantWidth = 1.5,
      noise = -40,
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
