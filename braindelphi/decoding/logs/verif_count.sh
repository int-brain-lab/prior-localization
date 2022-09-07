
COUNTER = 0
for file in slurm/*59341162*
do
        if cat ${file} | grep "not cached"
        then echo ${file}
        fi
        if cat ${file} | grep "not cached"
        then COUNTER=$(( COUNTER + 1 ))
        fi
done
printf "The value of the counter is COUNTER=%d\n" $COUNTER

# for file in slurm/*.out
# do
#         if ! (cat ${file} | grep "successful")
#         then echo ${file}
#         fi
# done

