Całość projektu zawarta jest w repozytorium. Dane wykorzystane do uzyskania wyników nie zostały zamieszczone z powodu ich wielkości.
W ramach projektu wykorzystano dwa zbiory danych:
- uzyskany przy pomocy sygnału Chirp
- uzyskany przy pomocy systemu samowzbudnego SAS
Dla każdego zbioru danych przeprowadzono analizę sygnału przy pomocy:
-PSD
-MFCC
-STFT
-transformaty falkowej
Dodatkowo dla sygnałów po operacji PSD i transformaty falkowej wykonano Principal Component Analysis.
W efekcie dało to 12 osobnych zbiorów danych wykorzystanych jako wsady do sieci neuronowych.
Wyszkolono ANN na danych PSD oraz transformaty falkowej(również dla PCA) oraz CNN dla danych MFCC i STFT
Dla wszystkich sieci neuronowych wykonano wykresy accuracy oraz loss procesu uczenia oraz przedstawiono confusion matrix działania.
Ponadto wykonano k-fold walidację ich działania.
Uzyskane wyniki zawarte są w folderach results oraz w folderze img (wykresy oraz confusion matrix). Nie zawarte zostały wyniki z konwolucyjnych sieci neuronowych działających na danych z systemu SAS. Powodem jest niewystarczająca ilość RAMu na wyszkolenie tych sieci na moim komputerze osobistym, jednakże kod potrzebny do wykononia tego zadania został zawarty w projekcie.