#!/bin/bash

echo touch all now ...

for dir in $(find -type d)
	do echo $dir; touch $dir/*;
done

