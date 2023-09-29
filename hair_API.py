import script.hair_mesh as HM
import script.hair_curves as HC
import os
import zlw


class HairTransformer:
    def __init__(self) -> None:
        self.tmp_dir = os.path.dirname(os.path.abspath(__file__))

    def testprompt(self, prompt):
        hair_style, color = self.hair_selector.prompt2index(prompt)
        print(hair_style)
        print(color)

    def name2mesh(self, hair_name, target_model_vs, output_path):
        """
        prompts: the prompts of hair style and color -> str
        target_model: the path of target head mesh -> str
        output_path: the path to save the hair -> str
        """
        neutral_model_path = fr"{self.tmp_dir}/head_model/neutral_model.obj"
        HM.transform(neutral_model_path, target_model_vs, fr"{self.tmp_dir}/hair_data/hair_mesh/{hair_name}.obj", output_path)


    def name2curves(self, hair_name, target_model_vs, output_path):
        """
        prompts: the prompts of hair style and color -> str
        target_model: the path of target head mesh -> str
        output_path: the path to save the hair -> str
        """
        neutral_model_path = fr"{self.tmp_dir}/head_model/neutral_model.obj"
        HC.transform(neutral_model_path, target_model_vs, fr"{self.tmp_dir}/hair_data/hair_curves/{hair_name}.abc", output_path)

        # input_path = fr"All_hair_abc/{hair_style}.abc"
        # self.transform(neutral_model, target_model, input_path, output_path, type="curves")


if __name__ == "__main__":
    tmp_dir = os.path.dirname(os.path.abspath(__file__))
    # print("+++",tmp_dir,"+++")
    target_model_vs = zlw.read_obj(fr"{tmp_dir}/head_model/head_new.obj", tri=True).vs
    HT = HairTransformer()
    for img_path in os.listdir("./hair_data/hair_mesh"):
        name = img_path[:-4]
        # if name != "woman_bun1.1":
        #     continue
        # HT.name2curves(fr"{name}",target_model_vs,fr"Out_hair_abc/{name}.abc")
        HT.name2mesh(fr"{name}",target_model_vs,fr"Out_hair_mesh/{name}.obj")
    
    
