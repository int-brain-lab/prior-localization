for file in slurm/*
do
        if cat ${file} | grep ERROR
        then echo ${file}
        fi
        if cat ${file} | grep Error
        then echo ${file}
        fi
        if cat ${file} | grep error
        then echo ${file}
        fi
done
