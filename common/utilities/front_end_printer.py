import os

class FrontEndPrinter():


    @staticmethod
    def write_training_process(project_id, run_id, epoch, reward, reward_of_sliding_window, property_result, output_file_path = "../front_end_output.txt"):
        f = open(output_file_path, 'w')
        try:
            f.write("PID: " + str(os.getpid()) + "\n")
            f.write("Project ID: " + str(project_id) + '\n')
            f.write("Run ID: " + str(run_id) + '\n')
            f.write("Epoch: " + str(epoch) + '\n')
            f.write("Reward: " +str(reward) + '\n')
            f.write("Reward Sliding Window: " + str(reward_of_sliding_window) + '\n')
            f.write("Property Result: " + str(property_result) + '\n')
        except:
            pass
        finally:
            f.close()

    @staticmethod
    def write_verification_result(project_id, run_id, property_query, property_result, output_file_path = "../front_end_output.txt"):
        f = open(output_file_path, 'w')
        try:
            f.write("Project ID: " + str(project_id) + '\n')
            f.write("Run ID: " + str(run_id) + '\n')
            f.write("Property Query: " + str(property_query) + '\n')
            f.write("Property Result: " + str(property_result) + '\n')
            f.close()
        except:
            pass
        finally:
            f.close()