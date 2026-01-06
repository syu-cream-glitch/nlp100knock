import torch
import torch.onnx as onnx
import torchvision.models as models

# モデルの重みの保存と読み込み
# PyTorchのモデルは学習したパラメータを内部に状態辞書（state_dict）として保持している．
# これらのパラメータ値をtorch.save()で永続化させられる．

model = models.vgg16(pretrained=True) # 事前学習済みモデルをロード
torch.save(model.state_dict(), 'model_weights.pth') # モデルの重みとバイアスのみを保存

# モデルの重みを読み込むためには，あらかじめ同じモデルの形をしたインスタンスを用意．
# インスタンスに対してload_state_dict()メソッドを使用し，パラメータ値を読み込む．

model = models.vgg16() # pretrained=Falseを引数に入れていないので，デフォルトのランダムな値
model.load_state_dict(torch.load('model_weights.pth'))
model.eval() # 評価モードに切り替え，DropoutやBatchNormなどの挙動を変更する．

# 注意：ドロップアウトやバッチノーマライゼーションレイヤーをevaluationモードに切り替えるために，推論前には model.eval()を実行する．
# これを忘れると，推論結果が正確じゃなくなる．

# モデルクラスの構造も一緒に保存したい場合
# torch.save(model, 'model.pth') # モデル全体を保存
# model = torch.load('model.pth') # モデル全体を読み込み

# onnx形式でのモデル出力
input_image = torch.zeros((1, 3, 224, 224)) # ダミー入力データ
onnx.export(model, input_image, 'model.onnx') # export
# onnxモデルを使用することで異なるプラットフォームや異なるプログラミング言語でディープラーニングモデルの推論を実行させるなど，様々なことが可能．
# 例えば，Caffe2やTensorFlowなどの他のフレームワークでモデルを使用したり，モバイルデバイスや組み込みシステムで推論を実行したりできる．
# model.onnxファイルが生成．