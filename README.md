# Výpočet Mandelbrotovej množiny

Tento repozitár obsahuje 2 možnosti výpočtu:

  - Paralelne (CUDA)
  - Sekvenčne ( C )

### Štruktúra projektu

  - /lib - Obsahuje knižnicu použitú na vykreslenie množiny.
  - /cuda - Obsahuje zdrojové kódy pre sekv. a paral. riešenie.
  - /src - Obsahuje Java kód pre generovanie výsledného obrázku.

### Spustenie

##### 1. Paralelný výpočet

Vyžaduje nainštalovanú CUDU, Visual Studio.

**Poznámka: súbor cl.exe musí byť pridaný do environment variables v PATHE.**
**Pre Visual Studio 2019 sa súbor nachádza na tejto ceste:**
`C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.23.28105\bin\Hostx64\x64`

Program sa spustí zadaním nasledujúcich príkazov:

```sh
cd cuda
nvcc mandelbrot_parallel.cu -o mandelbrot_parallel
./mandelbrot_parallel
```

Výsledkom je súbor output.txt z ktorého neskôr vygenerujeme obrázok množiny.

##### 2. Sekvenčný výpočet

Vyžaduje nainštalovaný gcc.
Program sa spustí zadaním nasledujúcich príkazov:

```sh
cd cuda
gcc mandelbrot_sequence.c
./a.out
```

Výsledkom je súbor output.txt z ktorého neskôr vygenerujeme obrázok množiny.

##### 3. Generovanie obrázka

Vyžaduje nainštalovanú javu, JDK 1.8 a Intellij IDEA.

**Poznámka: Pravdepobne bude treba pridať ručne knižnicu mandelbrot-vizualizer.jar z priečinku lib.**
**V IDEI sa to dá následovne: File --> Project Structure --> Libraries, kliknúť na + a Java**

Pre vygenerovanie obrázka je potrebné spustiť triedu Main.java v priečinku src/sk/tuke.

Výsledkom je súbor s názvom result.png
