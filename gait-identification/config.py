import os


class Project:
    project_path = os.path.dirname(os.path.abspath(__file__))
    test_data_path = "%s/data/test/" % project_path

    # the CASIA gait dataset B path, you can download the data from
    # CASIA website, the dirtory contains a lot sub dirtory
    # named such as 001,002...
    casia_dataset_b_path = "../DatasetB/silhouettes"

    casia_test_img = "%s/001/bg-01/090/001-bg-01-090-038.png" % casia_dataset_b_path

    casia_test_img_dir = "%s/004/nm-01/090/" % casia_dataset_b_path


    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path)

if __name__ == '__main__':
    print(Project.project_path)
