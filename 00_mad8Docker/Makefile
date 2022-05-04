
# Makefile for madpackage - check the paths for the libraries !

# 26-OCT-2004, M. Woodley (SLAC)
#    From /afs/cern.ch/user/h/hansg/public/www/src/mad_new/madpackage.tgz
# 01-NOV-2004, M. Woodley (SLAC)
#    Add SLAC variants of Linux and Sun builds; add "slac" ASTUCE flag to
#    "mad.standard.f" and "mad.standard.noplot.f" targets; define new target
#    "mad.windows" for extraction of source for Windows; remove obsolete
#    "x11" ASTUCE flag

# Linux (rhel4-32 at SLAC):
  SYS=linux
  FCOMP=gfortran
  FCOPTS=-Wno-globals -O3
  CERN=/usr/lib/i386-linux-gnu/
  Xlib=/usr/lib/i386-linux-gnu/
  LIBS=$(CERN)/libmathlib.a $(CERN)/libpacklib.a $(CERN)/libkernlib.a \
       $(Xlib)/libX11.a $(Xlib)/libxcb.a $(Xlib)/libXau.a $(Xlib)/libXdmcp.a -lnsl -lm -lc
  ASTLIB=
  GCC_FLAGS=-g -ansi -Wall -D_SWAP_ -D_LINUX_

# Solaris (at SLAC):
# SYS=sun
# FCOMP=f77
# CC=gcc
# FCOPTS= -O
# CERN=/cern/2002/lib
# Xlib=/usr/X/lib
# USR=/usr/lib
# LIBS=-L$(CERN) -L$(Xlib) -L$(USR) \
#      -lmathlib -lpacklib -lkernlib -lX11 -lnsl -lsocket -lm -lc
# ASTLIB=
# GCC_FLAGS=-ansi -Wall

# Linux (at CERN):
## FCOPTS=-Nx6400
# FCOPTS=-w
## f77, fort77, g77 or similar, possibly pgf77 - check it out
# FCOMP=g77
# CC=gcc
# CERN=/cern/2001/lib
# Xlib=/usr/X11R6/lib
# LIBS=$(CERN)/libmathlib.a $(CERN)/libpacklib.a $(CERN)/libkernlib.a $(Xlib)/libX11.a -lm -lc
# ASTLIB=
# SYS=linux
# GCC_FLAGS=-g -ansi -Wall -D_LINUX_

# HP (at CERN):
# FCOPTS=+ppu
# FCOMP=fort77
# CC=cc
# CERN=/cern/pro/lib
# Xlib=/usr/lib/X11R6
# LIBS=$(CERN)/libmathlib.a $(CERN)/libpacklib.a $(CERN)/libkernlib.a $(Xlib)/libX11.a -lm -lc
# ASTLIB=$(CERN)/libkernlib.a
# SYS=hp
# GCC_FLAGS=-ansi -Wall

# AIX (at CERN):
# FCOPTS=-qextname
# FCOPTS=-qextname -O2
# FCOMP=f77
# CC=cc
# CERN=/cern/pro/lib
# Xlib=/usr/lib
# LIBS=$(CERN)/libmathlib.a $(CERN)/libpacklib.a $(CERN)/libkernlib.a $(Xlib)/libX11.a -lm -lc
# ASTLIB=
# SYS=aix
# GCC_FLAGS=-ansi -Wall

# OSF1 (at CERN):
# FCOPTS=-O4 -v -tune ev5 -align dcommons -unroll 7
# FCOMP=f77
# CC=gcc
# CERN=/cern/pro/lib
# Xlib=/usr/lib
# LIBS=$(CERN)/libmathlib.a $(CERN)/libpacklib.a $(CERN)/libkernlib.a $(Xlib)/libX11.a -lm -lc
# ASTLIB=
# SYS=osf1
# GCC_FLAGS=-ansi -Wall -D_swap_

# the following SUN flags work at CERN
# SYS=sun
# FCOPTS= -s -O
# FCOMP=f77
# CC=gcc
# CERN=/cern/pro/lib
# Xlib=/usr/lib/X11
# USR=/usr/lib
# LIBS=-L$(CERN) -L$(Xlib) -L$(USR) -lmathlib -lpacklib -lkernlib -lsocket -lnsl -lX11
# ASTLIB=$(CERN)/libkernlib.a
# GCC_FLAGS=-ansi -Wall

astuce: astuce.f
	$(FCOMP) $(FCOPTS) -o astuce astuce.f $(ASTLIB)

doom.o: doom.c doom.h doomex.h
	$(CC) $(GCC_FLAGS) -c -o doom.o doom.c

doom_newdb: doom_newdb.c doom.o
	$(CC) $(GCC_FLAGS) -o doom_newdb doom_newdb.c doom.o -lm -lc

gxx11c.o: gxx11c.c
	$(CC) $(GCC_FLAGS) -c gxx11c.c

gxx11.o: gxx11.f
	$(FCOMP) -c $(FCOPTS) gxx11.f

gxx11.f: gxx11.ss
	astuce -s gxx11.ss -f gxx11.f -d fortran

gxx11c.c: gxx11.ss
	astuce -s gxx11.ss -f gxx11c.c -d c

mad.doom: mad.doom.o doom.o gxx11.o gxx11c.o
	$(FCOMP) $(FCOPTS) -o mad.doom mad.doom.o doom.o gxx11.o gxx11c.o $(LIBS) -lm -lc
	mv mad.doom mad8d

mad.doom.o: mad.doom.f
	$(FCOMP) $(FCOPTS) -c -o mad.doom.o mad.doom.f

mad.doom.f: mad8.ss
	astuce -s mad8.ss -f mad.doom.f -d $(SYS),unix,doom

mad.doom.noplot: mad.doom.noplot.o doom.o
	$(FCOMP) $(FCOPTS) -o mad.doom.noplot mad.doom.noplot.o doom.o $(LIBS) -lm -lc
	mv mad.doom.noplot mad8dnp

mad.doom.noplot.o: mad.doom.noplot.f
	$(FCOMP) $(FCOPTS) -c -o mad.doom.noplot.o mad.doom.noplot.f

mad.doom.noplot.f: mad8.ss
	astuce -s mad8.ss -f mad.doom.noplot.f -d unix,doom,noplot

mad.standard: mad.standard.o gxx11.o gxx11c.o
	$(FCOMP) $(FCOPTS) -o mad.standard mad.standard.o gxx11.o gxx11c.o $(LIBS) -lm -lc
	mv mad.standard mad8s

mad.standard.o: mad.standard.f
	$(FCOMP) $(FCOPTS) -c -o mad.standard.o mad.standard.f

mad.standard.f: mad8.ss
	astuce -s mad8.ss -f mad.standard.f -d unix,standard,slac

mad.standard.noplot: mad.standard.noplot.o
	$(FCOMP) $(FCOPTS) -o mad.standard.noplot mad.standard.noplot.o $(LIBS) -lm -lc
	mv mad.standard.noplot mad8snp

mad.standard.noplot.o: mad.standard.noplot.f
	$(FCOMP) $(FCOPTS) -c -o mad.standard.noplot.o mad.standard.noplot.f

mad.standard.noplot.f: mad8.ss
	astuce -s mad8.ss -f mad.standard.noplot.f -d unix,standard,slac,noplot

c6t: c6t.c doom.o
	$(CC) $(GCC_FLAGS) -o c6t c6t.c doom.o -lm -lc

SXF_in: SXF_in.c doom.o
	$(CC) $(GCC_FLAGS) -o SXF_in SXF_in.c doom.o -lm -lc

SXF_ex: SXF_ex.c doom.o
	$(CC) $(GCC_FLAGS) -o SXF_ex SXF_ex.c doom.o -lm -lc

get_corr3: get_corr3.f doom.o
	$(FCOMP) $(FCOPTS) -o get_corr3 get_corr3.f doom.o $(LIBS) -lm -lc

mad.windows: mad8.ss
	astuce -s mad8.ss -f mad.windows.f -d standard,slac,windows,comment
	astuce -s gxx11.ss -f gxx11.windows.f -d fortran,noterm,windows,comment
