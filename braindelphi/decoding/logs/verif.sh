for file in slurm/*61891657*
do
        if cat ${file} | grep ERROR
        then echo ${file}
        fi
        if cat ${file} | grep Error
        then echo ${file}
        fi
done


# for file in slurm/*.out
# do
#         if ! (cat ${file} | grep "successful")
#         then echo ${file}
#         fi
# done

