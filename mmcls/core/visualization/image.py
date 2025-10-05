# Copyright (c) OpenMMLab. All rights reserved.
import matplotlib.pyplot as plt
import mmcv
import numpy as np
from matplotlib.backend_bases import CloseEvent

# A small value
EPS = 1e-2


def color_val_matplotlib(color):
    """将各种BGR顺序的输入转换为标准化的RGB matplotlib颜色元组，

    参数：
        color (:obj:`mmcv.Color`/str/tuple/int/ndarray): 颜色输入

    返回：
        tuple[float]: 一个包含3个标准化浮点数的元组，表示RGB通道。
    """
    color = mmcv.color_val(color)
    color = [color / 255 for color in color[::-1]]
    return tuple(color)


class BaseFigureContextManager:
    """上下文管理器以重用 matplotlib 图形。

    它提供一个用于保存的图形和一个用于显示的图形，以支持
    不同的设置。

    参数：
        axis (bool)：是否显示坐标轴线。
        fig_save_cfg (dict)：用于保存图形的关键字参数。
            默认为空字典。
        fig_show_cfg (dict)：用于显示图形的关键字参数。
            默认为空字典。
    """

    def __init__(self, axis=False, fig_save_cfg={}, fig_show_cfg={}) -> None:
        self.is_inline = 'inline' in plt.get_backend()

        # 因为保存和显示需要不同的图形大小
        # 我们设置了两个图形和坐标轴来处理保存和显示
        self.fig_save: plt.Figure = None
        self.fig_save_cfg = fig_save_cfg
        self.ax_save: plt.Axes = None

        self.fig_show: plt.Figure = None
        self.fig_show_cfg = fig_show_cfg
        self.ax_show: plt.Axes = None

        self.axis = axis

    def __enter__(self):
        if not self.is_inline:
            # 如果使用内联后端，我们无法控制显示哪个图形，
            # 因此禁用交互式 fig_show，并将 fig_save 的初始化
            # 放入 `prepare` 函数中。
            self._initialize_fig_save()
            self._initialize_fig_show()
        return self

    def _initialize_fig_save(self):
        fig = plt.figure(**self.fig_save_cfg)
        ax = fig.add_subplot()

        # 通过设置子图边距去除白色边缘
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        self.fig_save, self.ax_save = fig, ax

    def _initialize_fig_show(self):
        # fig_save 将被调整为图像大小，只有 fig_show 需要 fig_size。
        fig = plt.figure(**self.fig_show_cfg)
        ax = fig.add_subplot()

        # 通过设置子图边距来去除白色边缘
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        self.fig_show, self.ax_show = fig, ax

    def __exit__(self, exc_type, exc_value, traceback):
        if self.is_inline:
            # 如果使用内联后端，是否关闭图形取决于用户是否想显示图像。
            return

        plt.close(self.fig_save)
        plt.close(self.fig_show)

    def prepare(self):
        if self.is_inline:
            # 如果使用内联后端，请重新构建 fig_save。
            self._initialize_fig_save()
            self.ax_save.cla()
            self.ax_save.axis(self.axis)
            return

        # 如果用户强制销毁窗口，请重新构建 fig_show。
        if not plt.fignum_exists(self.fig_show.number):
            self._initialize_fig_show()

        # Clear all axes
        self.ax_save.cla()
        self.ax_save.axis(self.axis)
        self.ax_show.cla()
        self.ax_show.axis(self.axis)

    def wait_continue(self, timeout=0, continue_key=' ') -> int:
        """显示图像并等待用户输入。

        此实现参考了
        https://github.com/matplotlib/matplotlib/blob/v3.5.x/lib/matplotlib/_blocking_input.py

        参数：
            timeout (int)：如果为正数，则在``timeout``秒后继续。
                默认为0。
            continue_key (str)：用户继续的按键。默认为
                空格键。

        返回：
            int：如果为零，表示超时或用户按下了``continue_key``，
                如果为一，表示用户关闭了显示图形。
        """  # noqa: E501
        if self.is_inline:
            # If use inline backend, interactive input and timeout is no use.
            return

        if self.fig_show.canvas.manager:
            # Ensure that the figure is shown
            self.fig_show.show()

        while True:

            # Connect the events to the handler function call.
            event = None

            def handler(ev):
                # Set external event variable
                nonlocal event
                # Qt 后端可能同时触发两个事件，
                # 使用条件来避免错过关闭事件。
                event = ev if not isinstance(event, CloseEvent) else event
                self.fig_show.canvas.stop_event_loop()

            cids = [
                self.fig_show.canvas.mpl_connect(name, handler)
                for name in ('key_press_event', 'close_event')
            ]

            try:
                self.fig_show.canvas.start_event_loop(timeout)
            finally:  # Run even on exception like ctrl-c.
                # Disconnect the callbacks.
                for cid in cids:
                    self.fig_show.canvas.mpl_disconnect(cid)

            if isinstance(event, CloseEvent):
                return 1  # Quit for close.
            elif event is None or event.key == continue_key:
                return 0  # Quit for continue.


class ImshowInfosContextManager(BaseFigureContextManager):
    """上下文管理器，用于重用 matplotlib 图形并在图像上放置信息。

    参数：
        fig_size (tuple[int]): 显示图像的图形大小。

    示例：
        >>> import mmcv
        >>> from mmcls.core import visualization as vis
        >>> img1 = mmcv.imread("./1.png")
        >>> info1 = {'class': 'cat', 'label': 0}
        >>> img2 = mmcv.imread("./2.png")
        >>> info2 = {'class': 'dog', 'label': 1}
        >>> with vis.ImshowInfosContextManager() as manager:
        ...     # 显示 img1
        ...     manager.put_img_infos(img1, info1)
        ...     # 在同一图形上显示 img2 并保存输出图像。
        ...     manager.put_img_infos(
        ...         img2, info2, out_file='./2_out.png')
    """

    def __init__(self, fig_size=(15, 10)):
        super().__init__(
            axis=False,
            # 适当的图像保存 dpi 和默认字体大小。
            fig_save_cfg=dict(frameon=False, dpi=36),
            fig_show_cfg=dict(frameon=False, figsize=fig_size))

    def _put_text(self, ax, text, x, y, text_color, font_size):
        ax.text(
            x,
            y,
            f'{text}',
            bbox={
                'facecolor': 'black',
                'alpha': 0.7,
                'pad': 0.2,
                'edgecolor': 'none',
                'boxstyle': 'round'
            },
            color=text_color,
            fontsize=font_size,
            family='monospace',
            verticalalignment='top',
            horizontalalignment='left')

    def put_img_infos(self,
                      img,
                      infos,
                      text_color='white',
                      font_size=26,
                      row_width=20,
                      win_name='',
                      show=True,
                      wait_time=0,
                      out_file=None):
        """显示带有额外信息的图像。

        参数：
            img (str | ndarray)：要显示的图像。
            infos (dict)：要在图像中显示的额外信息。
            text_color (:obj:`mmcv.Color`/str/tuple/int/ndarray)：额外信息的显示颜色。默认为'白色'。
            font_size (int)：额外信息的显示字体大小。默认为26。
            row_width (int)：图像上每行结果之间的宽度。
            win_name (str)：图像标题。默认为''。
            show (bool)：是否显示图像。默认为True。
            wait_time (int)：显示图像的秒数。默认为0。
            out_file (Optional[str])：写入图像的文件名。默认为None。

        返回：
            np.ndarray：带有额外信息的图像。
        """
        self.prepare()

        text_color = color_val_matplotlib(text_color)
        img = mmcv.imread(img).astype(np.uint8)

        x, y = 3, row_width // 2
        img = mmcv.bgr2rgb(img)
        width, height = img.shape[1], img.shape[0]
        img = np.ascontiguousarray(img)

        # 添加一个小的EPS以避免因matplotlib的
        # 截断而导致的精度丢失 (https://github.com/matplotlib/matplotlib/issues/15363)
        dpi = self.fig_save.get_dpi()
        self.fig_save.set_size_inches((width + EPS) / dpi,
                                      (height + EPS) / dpi)

        for k, v in infos.items():
            if isinstance(v, float):
                v = f'{v:.2f}'
            label_text = f'{k}: {v}'
            self._put_text(self.ax_save, label_text, x, y, text_color,
                           font_size)
            if show and not self.is_inline:
                self._put_text(self.ax_show, label_text, x, y, text_color,
                               font_size)
            y += row_width

        self.ax_save.imshow(img)
        stream, _ = self.fig_save.canvas.print_to_buffer()
        buffer = np.frombuffer(stream, dtype='uint8')
        img_rgba = buffer.reshape(height, width, 4)
        rgb, _ = np.split(img_rgba, [3], axis=2)
        img_save = rgb.astype('uint8')
        img_save = mmcv.rgb2bgr(img_save)

        if out_file is not None:
            mmcv.imwrite(img_save, out_file)

        ret = 0
        if show and not self.is_inline:
            # Reserve some space for the tip.
            self.ax_show.set_title(win_name)
            self.ax_show.set_ylim(height + 20)
            self.ax_show.text(
                width // 2,
                height + 18,
                'Press SPACE to continue.',
                ha='center',
                fontsize=font_size)
            self.ax_show.imshow(img)

            # Refresh canvas, necessary for Qt5 backend.
            self.fig_show.canvas.draw()

            ret = self.wait_continue(timeout=wait_time)
        elif (not show) and self.is_inline:
            # 如果使用内联后端，我们使用 fig_save 来显示图像
            # 所以如果用户不想显示，我们需要关闭它。
            plt.close(self.fig_save)

        return ret, img_save


def imshow_infos(img,
                 infos,
                 text_color='white',
                 font_size=26,
                 row_width=20,
                 win_name='',
                 show=True,
                 fig_size=(15, 10),
                 wait_time=0,
                 out_file=None):
    """显示带有额外信息的图像。

    参数：
        img (str | ndarray)：要显示的图像。
        infos (dict)：要在图像中显示的额外信息。
        text_color (:obj:`mmcv.Color`/str/tuple/int/ndarray)：额外信息的
            显示颜色。默认为“白色”。
        font_size (int)：额外信息显示的字体大小。默认为26。
        row_width (int)：图像上每行结果之间的宽度。
        win_name (str)：图像标题。默认为“”。
        show (bool)：是否显示图像。默认为True。
        fig_size (tuple)：图像显示的图形大小。默认为（15, 10）。
        wait_time (int)：显示图像的秒数。默认为0。
        out_file (可选[str])：写入图像的文件名。
            默认为None。

    返回：
        np.ndarray：带有额外信息的图像。
    """
    with ImshowInfosContextManager(fig_size=fig_size) as manager:
        _, img = manager.put_img_infos(
            img,
            infos,
            text_color=text_color,
            font_size=font_size,
            row_width=row_width,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)
    return img
