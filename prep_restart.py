#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# written by Christoph Federrath, 2014-2018

import os
import sys
import time
import array
import re
import subprocess
import fnmatch
from tempfile import mkstemp
from shutil import move, copyfile
from os import remove, close, path

def read_from_checkpointfile(chkfilename, dset, param):
	debug = False
	return_val = 0
	fh, tempfile = mkstemp()
	ftemp = open(tempfile, 'w')
	shellcmd = "h5dump -d \""+dset+"\" "+chkfilename+" |grep '\""+param+"' -A 1"
	#print shellcmd
	subprocess.call(shellcmd, shell=True, stdout=ftemp, stderr=ftemp)
	ftemp.close()
	close(fh)
	ftemp = open(tempfile, 'r')
	for line in ftemp:
		if line.find(param)!=-1: continue
		if (dset=="integer scalars") or (dset=="integer runtime parameters"):
			return_val = int(line.lstrip().rstrip())
		if (dset=="real scalars") or (dset=="real runtime parameters"):
			return_val = float(line.lstrip().rstrip())
	remove(tempfile)
	if debug==True: print param+" = ", return_val
	return return_val


def get_checkpointfilenumber(chkfilename):
	debug = False
	if debug==True: print "reading checkpointfilenumber from "+chkfilename+"..."
	return_val = read_from_checkpointfile(chkfilename, "integer scalars", "checkpointfilenumber")
	if debug==True: print "checkpointfilenumber = ", return_val
	return return_val

def get_next_plotfilenumber(chkfilename):
	debug = False
	if debug==True: print "reading plotfilenumber from "+chkfilename+"..."
	return_val = read_from_checkpointfile(chkfilename, "integer scalars", "plotfilenumber")
	if debug==True: print "next plotfilenumber = ", return_val
	return return_val

def get_next_particlefilenumber(chkfilename):
	debug = False
	if debug==True: print "reading particlefilenumber from "+chkfilename+"..."
	return_val = read_from_checkpointfile(chkfilename, "integer scalars", "particlefilenumber")
	if debug==True: print "next particlefilenumber = ", return_val
	return return_val

def get_next_moviefilenumber(chkfilename):
	debug = False
	if debug==True: print "computing next movie file number from "+chkfilename+"..."
	try:
		movie_dump_num = read_from_checkpointfile(chkfilename, "integer runtime parameters", "movie_dump_num")
	except ValueError:
		print "Movie module was not compiled in. Proceeding without..."
		return -1
	movie_dstep_dump = read_from_checkpointfile(chkfilename, "integer runtime parameters", "movie_dstep_dump")
	movie_dt_dump = read_from_checkpointfile(chkfilename, "real runtime parameters", "movie_dt_dump")
	simtime = read_from_checkpointfile(chkfilename, "real scalars", "time")
	simstep = read_from_checkpointfile(chkfilename, "integer scalars", "nstep")
	if debug==True: print "movie_dstep_dump, movie_dt_dump = ", movie_dstep_dump, movie_dt_dump
	if movie_dstep_dump==0 and movie_dt_dump==0.0:
		return -1
	if movie_dstep_dump==0:
		return_val = int(simtime/movie_dt_dump)+1
	if movie_dt_dump==0.0:
		return_val = int(simstep/movie_dstep_dump)+1
	if movie_dump_num!=return_val:
		print "CAUTION: movie_dump_num in chk file is NOT equal to computed next movie_dump_num! Using chk movie_dump_num anyway."
	return_val = movie_dump_num
	if debug==True: print "next movie_dump_num = ", return_val
	return return_val

def inplace_change_flash_par(filename, restart, chknum, pltnum, partnum, movnum):
	debug = True
	fh, tempfile = mkstemp()
	ftemp = open(tempfile, 'w')
	f = open(filename, 'r')
	for line in f:
		# replace restart
		if line.lower().find("restart")==0:
			if debug==True: print filename+": found line   : "+line.rstrip()
			i = line.find("=")
			newline = line[0:i+1]+" "+restart+"\n"
			line = newline
			if debug==True: print filename+": replaced with: "+line.rstrip()
		# replace checkpointfilenumber
		if line.lower().find("checkpointfilenumber")==0:
			if debug==True: print filename+": found line   : "+line.rstrip()
			i = line.find("=")
			newline = line[0:i+1]+" %(#)d\n" % {"#":chknum}
			line = newline
			if debug==True: print filename+": replaced with: "+line.rstrip()
		# replace plotfilenumber
		if line.lower().find("plotfilenumber")==0:
			if debug==True: print filename+": found line   : "+line.rstrip()
			i = line.find("=")
			newline = line[0:i+1]+" %(#)d\n" % {"#":pltnum}
			line = newline
			if debug==True: print filename+": replaced with: "+line.rstrip()
		# replace particlefilenumber
		if line.lower().find("particlefilenumber")==0:
			if debug==True: print filename+": found line   : "+line.rstrip()
			i = line.find("=")
			newline = line[0:i+1]+" %(#)d\n" % {"#":partnum}
			line = newline
			if debug==True: print filename+": replaced with: "+line.rstrip()
		# replace movie_dump_num
		if line.lower().find("movie_dump_num")==0:
			if debug==True: print filename+": found line   : "+line.rstrip()
			i = line.find("=")
			newline = line[0:i+1]+" %(#)d\n" % {"#":movnum}
			line = newline
			if debug==True: print filename+": replaced with: "+line.rstrip()
                # replace init_restart with .false. (if present)
                if line.lower().find("init_restart")==0:
                        if debug==True: print filename+": found line   : "+line.rstrip()
                        i = line.find("=")
                        newline = line[0:i+1]+".false.\n"
                        line = newline
                        if debug==True: print filename+": replaced with: "+line.rstrip()
		# add lines to temporary output file
		ftemp.write(line)
	ftemp.close()
	close(fh)
	f.close()
	remove(filename)
	move(tempfile, filename)
        os.chmod(filename, 0644)

def get_job_file_name():
	if os.path.isfile("job.cmd"):
		print "Found job.cmd. Replacing shell.out..."
		return "job.cmd"
	if os.path.isfile("job.sh"):
		print "Found job.sh. Replacing shell.out..."
		return "job.sh"

def inplace_change_job_file(filename, newflag):
	debug = True
	fh, tempfile = mkstemp()
	ftemp = open(tempfile, 'w')
	f = open(filename, 'r')
	for line in f:
		# increment shell*.out
		if line.find("shell.out")>0:
			if debug==True: print filename+": found line   : "+line.rstrip()
			ibeg = line.find("shell.out")+9
			iend = line.find(" 2>&1")
			count_string = line[ibeg:iend].rstrip()
			if count_string == "":
				count = 0
			else:
				count = int(count_string)
			count += 1
			if newflag==True: count = 0
			new_count_string = "%(#)02d" % {"#":count}
			newline = line[0:ibeg]+new_count_string+line[iend:len(line)]
			line = newline
			if debug==True: print filename+": replaced with: "+line.rstrip()
		# add lines to temporary output file
		ftemp.write(line)
	ftemp.close()
	close(fh)
	f.close()
	remove(filename)
	move(tempfile, filename)
	os.chmod(filename, 0644)

def help_me(args, nargs):
	for arg in args:
		if ((nargs < 2) or arg.find("-help")!=-1) or (arg.find("--help")!=-1):
			print
			print "USAGE options for "+args[0]+":"
			print " "+args[0]+" <filename> (filename must be a FLASH chk file for restart)"
			print " "+args[0]+" -auto      (uses last available chk file in current directory)"
			print " "+args[0]+" -new       (prepares new simulation: restart = .false.)"
			print
			quit()


# ===== MAIN Start =====

args = sys.argv
nargs = len(sys.argv)
help_me(args, nargs)

# make a backup copy of flash.par
print "Copying 'flash.par' to 'flash.par_restart_backup' as backup."
copyfile("flash.par","flash.par_restart_backup")

# reset flash.par for new simulations (restart = .false.)
if args[1] == "-new":
	print "Resetting flash.par for starting FLASH from scratch (restart = .false.):"
	inplace_change_flash_par("flash.par", ".false.", 0, 0, 0, 0)
	job_file = get_job_file_name()
	if job_file: inplace_change_job_file(job_file, True)
	quit()

# automatically determine last chk file and prepare restart = .true.
if args[1] == "-auto":
	chkfiles = []
	for file in os.listdir('.'):
		if fnmatch.fnmatch(file, '*_hdf5_chk_*'):
			chkfiles.append(file)
	chkfiles.sort()
	chkfilename = chkfiles[len(chkfiles)-1]
	print "Found the following last FLASH checkpoint file: '"+chkfilename+"'"
	print "...and using it to prepare restart:"
	chknum = get_checkpointfilenumber(chkfilename)
	pltnum = get_next_plotfilenumber(chkfilename)
	partnum = get_next_particlefilenumber(chkfilename)
	movnum = get_next_moviefilenumber(chkfilename)
	inplace_change_flash_par("flash.par", ".true.", chknum, pltnum, partnum, movnum)
	job_file = get_job_file_name()
	if job_file: inplace_change_job_file(job_file, False)
	quit()

# assume that user supplied the name of a valid chk file
chkfilename = args[1]
print "Using FLASH checkpoint file '"+chkfilename+"' to prepare restart:"
chknum = get_checkpointfilenumber(chkfilename)
pltnum = get_next_plotfilenumber(chkfilename)
partnum = get_next_particlefilenumber(chkfilename)
movnum = get_next_moviefilenumber(chkfilename)
inplace_change_flash_par("flash.par", ".true.", chknum, pltnum, partnum, movnum)
job_file = get_job_file_name()
if job_file: inplace_change_job_file(job_file, False)

# ===== MAIN End =====
