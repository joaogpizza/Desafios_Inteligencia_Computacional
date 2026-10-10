import os

# Definindo a raiz a partir do diretório atual do script (subindo um nível)
PATH_RAIZ = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

PATH_DATA = os.path.join(PATH_RAIZ, "data")
PATH_SCRIPTS = os.path.join(PATH_RAIZ, "scripts")
PATH_RESULTS = os.path.join(PATH_RAIZ, "results")

PATH_RAW = os.path.join(PATH_DATA, "raw")
PATH_WHU = os.path.join(PATH_RAW, "WHU")

PATH_RAW_TRAIN = os.path.join(PATH_WHU, "train")
PATH_RAW_TRAIN_IMG = os.path.join(PATH_RAW_TRAIN, "Image")
PATH_RAW_TRAIN_MASK = os.path.join(PATH_RAW_TRAIN, "Mask")

PATH_RAW_VAL = os.path.join(PATH_WHU, "val")
PATH_RAW_VAL_IMG = os.path.join(PATH_RAW_VAL, "Image")
PATH_RAW_VAL_MASK = os.path.join(PATH_RAW_VAL, "Mask")

PATH_RAW_TEST = os.path.join(PATH_WHU, "test")
PATH_RAW_TEST_IMG = os.path.join(PATH_RAW_TEST, "Image")
PATH_RAW_TEST_MASK = os.path.join(PATH_RAW_TEST, "Mask")

PATH_PATCHES_IMGS = os.path.join(PATH_DATA, "patches_imgs")
PATH_PATCHES_IMGS_TRAIN = os.path.join(PATH_PATCHES_IMGS, "train")
PATH_PATCHES_IMGS_VAL = os.path.join(PATH_PATCHES_IMGS, "val")
PATH_PATCHES_IMGS_TEST = os.path.join(PATH_PATCHES_IMGS, "test")

PATH_PATCHES_GT = os.path.join(PATH_DATA, "patches_gt")
PATH_PATCHES_GT_TRAIN = os.path.join(PATH_PATCHES_GT, "train")
PATH_PATCHES_GT_VAL = os.path.join(PATH_PATCHES_GT, "val")
PATH_PATCHES_GT_TEST = os.path.join(PATH_PATCHES_GT, "test")

PATH_PLOTS = os.path.join(PATH_RESULTS, "plots")
PATH_PLOTS_CURVAS_APRENDIZADO = os.path.join(PATH_PLOTS, "curvas_aprendizado")
PATH_PLOTS_CURVA_APRENDIZADO_LOSS = os.path.join(
    PATH_PLOTS_CURVAS_APRENDIZADO,
    "loss"
)
PATH_PLOTS_CURVA_APRENDIZADO_MDICE = os.path.join(
    PATH_PLOTS_CURVAS_APRENDIZADO,
    "mdice"
)
PATH_PLOTS_CURVA_APRENDIZADO_TAXA_APRENDIZADO = os.path.join(
    PATH_PLOTS_CURVAS_APRENDIZADO,
    "taxa_aprendizado"
)
