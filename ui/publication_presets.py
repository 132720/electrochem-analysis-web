PUBLICATION_PRESETS = {

    "通用 SCI 推荐": {

        "sizes": [
            (
                "单栏（90 mm）",
                90.0,
                70.0,
            ),
            (
                "1.5 栏（140 mm）",
                140.0,
                95.0,
            ),
            (
                "双栏（180 mm）",
                180.0,
                120.0,
            ),
        ],

        "font_family":
            "Arial",

        "label_font_size":
            9,

        "tick_font_size":
            8,

        "legend_font_size":
            8,

        "line_width":
            1.2,

        "axis_line_width":
            0.8,

        "marker_size":
            4.0,

        "dpi":
            1200,

        "note":
            "",
    },


    "爱思唯尔（Elsevier）": {

        "sizes": [
            (
                "单栏（90 mm）",
                90.0,
                70.0,
            ),
            (
                "1.5 栏（140 mm）",
                140.0,
                95.0,
            ),
            (
                "双栏（190 mm）",
                190.0,
                125.0,
            ),
        ],

        "font_family":
            "Arial",

        "label_font_size":
            9,

        "tick_font_size":
            8,

        "legend_font_size":
            8,

        "line_width":
            1.0,

        "axis_line_width":
            0.8,

        "marker_size":
            4.0,

        "dpi":
            1000,

        "note":
            (
                "Elsevier 当前图稿规范："
                "单栏 90 mm、1.5 栏 140 mm、双栏 190 mm；"
                "线图位图通常采用 1000 dpi，"
                "特别精细的线图可使用更高分辨率。"
            ),
    },


    "施普林格·自然（Springer Nature，通用参考）": {

        "sizes": [
            (
                "单栏参考（90 mm）",
                90.0,
                70.0,
            ),
            (
                "中等宽度参考（140 mm）",
                140.0,
                95.0,
            ),
            (
                "双栏参考（180 mm）",
                180.0,
                120.0,
            ),
        ],

        "font_family":
            "Arial",

        "label_font_size":
            9,

        "tick_font_size":
            8,

        "legend_font_size":
            8,

        "line_width":
            1.0,

        "axis_line_width":
            0.8,

        "marker_size":
            4.0,

        "dpi":
            1200,

        "note":
            (
                "Springer Nature 不同期刊的最终图稿规格可能不同；"
                "这里采用保守的通用线图参考值，"
                "投稿前请核对目标期刊。"
            ),
    },


    "自然（Nature）": {

        "sizes": [
            (
                "单栏（89 mm）",
                89.0,
                68.0,
            ),
            (
                "1.5 栏（128 mm）",
                128.0,
                90.0,
            ),
            (
                "双栏（183 mm）",
                183.0,
                118.0,
            ),
        ],

        "font_family":
            "Arial",

        "label_font_size":
            7,

        "tick_font_size":
            6,

        "legend_font_size":
            6,

        "line_width":
            0.8,

        "axis_line_width":
            0.7,

        "marker_size":
            3.5,

        "dpi":
            600,

        "note":
            (
                "Nature 最终图通常优先提交可编辑矢量图；"
                "主图常用 89 mm 单栏或 183 mm 双栏，"
                "普通图中文字约 5–7 pt。"
            ),
    },
}


DPI_OPTIONS = [
    300,
    600,
    1000,
    1200,
]


def get_preset(
    name: str,
) -> dict:

    return PUBLICATION_PRESETS[
        name
    ]
