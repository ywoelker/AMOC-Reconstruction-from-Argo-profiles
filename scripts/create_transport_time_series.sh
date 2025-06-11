# for cycle in '1st_7024' '2nd_5824' '3rd_5824' '4th_5824' '5th_5824' '6th_5824'; do
    
#     ## make a for loop of the strings ['90D', '365D', '1825D']
#     for time_smooth in '90' '365' '1825'; do
#     # for time_smooth in '10' '30'; do

#         for reference_level in -4800 -2000; do

#             echo ${cycle} ${time_smooth} ${reference_level};
#             python data_scripts/calc_transport_from_moorings.py --mooring_group_sim datasets/mooring_data/mooring_groups_${cycle}.nc --ts_gridded datasets/ts_gridded.nc --time_mapping_dict datasets/smoothing_${time_smooth}_days/argo_after_2012/paperdraft/time_mapping_dict_${cycle}.pickle --cycle_name KFS003-${cycle} --reference_level ${reference_level} --time_smooth ${time_smooth}D;

#         done

#     done

# done

base_path="../data_publication/datasets"


for cycle in '1st_7024' '2nd_5824' '3rd_5824' '4th_5824' '5th_5824' '6th_5824'; do
    
    ## make a for loop of the strings ['90D', '365D', '1825D']
    for time_smooth in '90'; do
    # for time_smooth in '10' '30'; do

        for reference_level in -4800; do

            echo ${cycle} ${time_smooth} ${reference_level};
            python -m amoc_reconstruction.preprocessing.calc_transport_from_moorings --mooring_group_sim ${base_path}/mooring_data/mooring_groups_${cycle}.nc --ts_gridded ${base_path}/ts_gridded.nc --time_mapping_dict ${base_path}/smoothing_${time_smooth}_days/argo_after_2012/paperdraft/time_mapping_dict_${cycle}.pickle --cycle_name KFS003-${cycle} --reference_level ${reference_level} --time_smooth ${time_smooth}D;

        done

    done

done
