# ベースイメージをUbuntu 22.04に更新（flash_attnのため）
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# ビルド時の非対話モード設定
ENV TZ=Asia/Tokyo
ENV DEBIAN_FRONTEND=noninteractive

# システムパッケージの更新とインストール
RUN apt-get update && apt-get install -y \
    tzdata wget git curl sudo bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Minicondaをインストール
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

# PATHを永続的に設定
ENV PATH=$CONDA_DIR/bin:$PATH

# --- ここからが改善点 ---

# 作業ディレクトリを/workspaceに設定（より一般的）
WORKDIR /workspace

# Conda環境構築用のYAMLをコピー
COPY environment.yml .

# Anacondaリポジトリの利用規約に同意する
RUN conda config --set always_yes yes --set changeps1 no && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# YAMLファイルからConda環境を作成し、キャッシュをクリーンアップ
# これで、環境に必要なライブラリが全てインストールされる
RUN conda env create -f environment.yml && \
    conda clean --all -y

# flash-attnを特別なオプション付きで個別にインストール
RUN conda run -n crystal_ds_python3.10_qlora pip install flash-attn==2.5.6 --no-build-isolation

# 今後のRUN/CMD命令で、デフォルトでこのConda環境を使うように設定
SHELL ["conda", "run", "-n", "crystal_ds_python3.10_qlora", "/bin/bash", "-c"]

# ユーザー`an`を作成し、sudo権限を付与
# ここでホストのUID/GIDを指定する
RUN groupadd -g 1100 M1 && \
    useradd -m -s /bin/bash -u 1105 -g 1100 an && \
    adduser an sudo

# 作業ディレクトリの所有者をanに変更
# このRUN命令はrootとして実行される
RUN chown -R an:M1 /workspace

# ユーザーを`an`に切り替え
USER an

# コンテナ起動時のデフォルトコマンド
CMD ["/bin/bash"]