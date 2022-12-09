# Taxi
#python cool_mc.py --parent_run_id=a06c920b73be4015ad456011d60ffdd8 --task=rl_model_checking --project_name="experiments" --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmin=? [F jobs_done=2]" --permissive_input="robustness,1"
#python cool_mc.py --parent_run_id=a06c920b73be4015ad456011d60ffdd8 --task=rl_model_checking --project_name="experiments" --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmin=? [ fuel>=4 U (G jobs_done = 1 & \"not_empty\") ]" --permissive_input="robustness,1,x"
#python cool_mc.py --parent_run_id=a06c920b73be4015ad456011d60ffdd8 --task=rl_model_checking --project_name="experiments" --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmin=? [ (((jobs_done=0 U (x=1 & y=2)) U (jobs_done=0 & !(x=1 & y=2))) U (\"empty\" & jobs_done=0)) ]" --permissive_input="robustness,1,x"
#python cool_mc.py --parent_run_id=a06c920b73be4015ad456011d60ffdd8 --task=rl_model_checking --project_name="experiments" --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmin=? [F \"empty\"]" --permissive_input="robustness,1,done"
#python cool_mc.py --parent_run_id=a06c920b73be4015ad456011d60ffdd8 --task=rl_model_checking --project_name="experiments" --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmin=? [F jobs_done=2]" --permissive_input="robustness,1,done"
#python cool_mc.py --parent_run_id=a06c920b73be4015ad456011d60ffdd8 --task=rl_model_checking --project_name="experiments" --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmin=? [F jobs_done=2]" --permissive_input="robustness,1,fuel"
#python cool_mc.py --parent_run_id=a06c920b73be4015ad456011d60ffdd8 --task=rl_model_checking --project_name="experiments" --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmin=? [F \"empty\"]" --permissive_input="robustness,1,passenger_loc_x"
#python cool_mc.py --parent_run_id=a06c920b73be4015ad456011d60ffdd8 --task=rl_model_checking --project_name="experiments" --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmin=? [F \"empty\"]" --permissive_input="robustness,1,passenger_dest_y"
# Freeway
#python cool_mc.py --parent_run_id=eb228d45f182499d985dc88480fa6315 --task=rl_model_checking --project_name="experiments" --prism_file_path="freeway.prism" --constant_definitions="" --prop="Pmin=? [F won=true]" --permissive_input="robustness,1,px0+px1+px2+px3"
#python cool_mc.py --parent_run_id=eb228d45f182499d985dc88480fa6315 --task=rl_model_checking --project_name="experiments" --prism_file_path="freeway.prism" --constant_definitions="" --prop="Pmin=? [F won=true]" --permissive_input="robustness,1,px0"
# Collision 1
#python cool_mc.py --parent_run_id=1a8fbc040431460989305ae5f72d69ca --task=rl_model_checking --project_name="experiments" --prism_file_path="avoid_obs1.prism" --constant_definitions="xMax=4,yMax=4,slickness=0" --prop="Pmin=? [F<=100 COLLISION=true]" --permissive_input="robustness,1"
# Smart Grid
#python cool_mc.py --parent_run_id=09daacf1cfd94bed83c8e7fb969ad996 --task=rl_model_checking --project_name="experiments" --prism_file_path="smart_grid.prism" --constant_definitions="max_consumption=4,renewable_limit=3,non_renewable_limit=3,grid_upper_bound=2" --prop="Pmax=? [F<=100 IS_BLACKOUT=true]" --permissive_input="robustness,1,non_renewable"
# Stock Market
#python cool_mc.py --parent_run_id=2ff1fb06d338455f8508b0c837e66039 --task=rl_model_checking --project_name="experiments" --prism_file_path="stock_market.prism" --constant_definitions="" --prop="Pmax=? [F \"bankruptcy\"]" --permissive_input="robustness,1,sell_price"
#python cool_mc.py --parent_run_id=2ff1fb06d338455f8508b0c837e66039 --task=rl_model_checking --project_name="experiments" --prism_file_path="stock_market.prism" --constant_definitions="" --prop="Pmax=? [F \"bankruptcy\"]" --permissive_input="robustness,1,sell_price"

# Freeway
#python cool_mc.py --parent_run_id=eb228d45f182499d985dc88480fa6315 --task=rl_model_checking --project_name="experiments" --prism_file_path="freeway.prism" --constant_definitions="" --prop="Pmin=? [F won=true]" --permissive_input="robustness,1,px0+px1+px2+px3"
# Collision 1
#python cool_mc.py --parent_run_id=1a8fbc040431460989305ae5f72d69ca --task=rl_model_checking --project_name="experiments" --prism_file_path="avoid_obs1.prism" --constant_definitions="xMax=4,yMax=4,slickness=0" --prop="Pmin=? [F<=100 COLLISION=true]" --permissive_input="robustness,1,obstacle1_x"


# Figure 4 robustness

#python cool_mc.py --parent_run_id=a06c920b73be4015ad456011d60ffdd8 --task=rl_model_checking --project_name="experiments" --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmin=? [ (!(x=1 & y=2) U (jobs_done=1 & !(x=1 & y=2))) U (\"empty\" & jobs_done=1)]" --permissive_input="robustness,1,x" 

#python cool_mc.py --parent_run_id=a06c920b73be4015ad456011d60ffdd8 --task=rl_model_checking --project_name="experiments" --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmin=? [ fuel>=4 U (G (jobs_done=1 & \"not_empty\"))]" --permissive_input="robustness,1,done" 
#python cool_mc.py --parent_run_id=a06c920b73be4015ad456011d60ffdd8 --task=rl_model_checking --project_name="experiments" --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmin=? [jobs_done=0 U (\"empty\" & passenger=true)]" --permissive_input="robustness,2,fuel"
#python cool_mc.py --parent_run_id=a06c920b73be4015ad456011d60ffdd8 --task=rl_model_checking --project_name="experiments" --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmin=? [jobs_done<2 U (\"empty\" & passenger=true)]" --permissive_input="robustness,2,x"
#python cool_mc.py --parent_run_id=a06c920b73be4015ad456011d60ffdd8 --task=rl_model_checking --project_name="experiments" --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmin=? [jobs_done<2 U (\"empty\" & passenger=false)]" --permissive_input="robustness,1,fuel"
#python cool_mc.py --parent_run_id=a06c920b73be4015ad456011d60ffdd8 --task=rl_model_checking --project_name="experiments" --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmin=? [jobs_done<2 U (\"empty\" & passenger=false)]" --permissive_input="robustness,1,x" 
#python cool_mc.py --parent_run_id=a06c920b73be4015ad456011d60ffdd8 --task=rl_model_checking --project_name="experiments" --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmin=? [jobs_done<2 U (\"empty\" & passenger=false)]" --permissive_input="robustness,1,passenger_loc_x" 