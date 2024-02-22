(** "micrograd" autograd engine based on the tutorial video by Andrej
    Karpathy: https://youtu.be/VMj-3S1tku0?si=6zWb3OfCKiu_UjCA. See
    also the main micrograd repo:
    https://github.com/karpathy/micrograd *)

open Core

(** Differentiable expressions. *)
type exp =
  | Num of float
  | Var of string
  | Add of exp * exp
  | Mul of exp * exp
  | Exp of exp
  | Pow of exp * float
                   [@@deriving sexp, compare]

(* let rec sexp_of_exp (v : exp) : Sexp.t = *)
(*   match v with *)
(*   | Num n -> Sexp.Atom (string_of_float n) *)
(*   | Var x -> Sexp.Atom x *)
(*   | Add (l, r) -> Sexp.List [sexp_of_exp l; Sexp.Atom "+"; sexp_of_exp r] *)
(*   | Mul (l, r) -> Sexp.List [sexp_of_exp l; Sexp.Atom "*"; sexp_of_exp r] *)
(*   | Exp e -> Sexp.List [Sexp.Atom "exp"; sexp_of_exp e] *)
(*   | Pow (e, k) -> Sexp.List [Sexp.Atom "pow"; sexp_of_exp e; *)
(*                              Sexp.Atom (string_of_float k)] *)

module S = Set.Make(String)

(** Free variables of an expression. *)
let rec fvs (e : exp) : S.t =
  match e with
  | Num _ -> S.empty
  | Var x -> S.singleton x
  | Add (l, r) -> Set.union (fvs l) (fvs r)
  | Mul (l, r) -> Set.union (fvs l) (fvs r)
  | Exp e -> fvs e
  | Pow (e, _) -> fvs e

(** Curried constructors and derived forms. *)
let num n = Num n
let var x = Var x
let add l r = Add (l, r)
let mul l r = Mul (l, r)
let exp e = Exp e
let pow e k = Pow (e, k)
let div num denom = mul num @@ pow denom (-1.0)
let neg e = mul e @@ num (-1.0)
let sub l r = add l @@ neg r
let tanh e = div
               (sub (exp @@ mul (num 2.0) e) (num 1.0))
               (add (exp @@ mul (num 2.0) e) (num 1.0))

(** Sum a list of expressions. *)
let sum (es : exp list) : exp =
  List.fold es ~init:(Num 0.0) ~f:add

(* let string_of_exp (v : exp) : string = *)
(*   Sexp.to_string @@ sexp_of_exp v *)

(** We use variable environments to assign values to variables. *)
module Env = Map.Make(String)

(** Evaluate an expression in a given environment. *)
let rec eval (env : float Env.t) (e : exp) : float =
  match e with
  | Num x -> x
  | Var x -> Map.find_exn env x
  | Add (l, r) -> eval env l +. eval env r
  | Mul (l, r) -> eval env l *. eval env r
  | Exp e -> Float.exp @@ eval env e
  | Pow (e, k) -> eval env e **. k

(** The backpropagation function produces a [Grad] object that maps
    each variable in the expression to the partial derivative of the
    root expression wrt. that variable. *)
module Grad = Map.Make(String)

(** Calculate partial derivatives of root expression wrt. all
    variables inside it. *)
let backprop (env : float Env.t) (root : exp) : float Grad.t =
  let rec go (e : exp) (deriv : float) : float Grad.t =
    match e with
    | Num _ -> Grad.empty
    | Var x -> Map.set Grad.empty ~key:x ~data:deriv
    | Add (l, r) ->
       let gl = go l deriv in
       let gr = go r deriv in
       Map.fold gl ~init:gr
         ~f:(fun ~key:k ~data:d acc -> Map.update acc k ~f:(fun o ->
                                           match o with
                                           | None -> d
                                           | Some x -> x +. d))
    | Mul (l, r) ->
       let gl = go l @@ eval env r *. deriv in
       let gr = go r @@ eval env l *. deriv in
       Map.fold gl ~init:gr
         ~f:(fun ~key:k ~data:d acc -> Map.update acc k ~f:(fun o ->
                                           match o with
                                           | None -> d
                                           | Some x -> x +. d))
    | Exp e ->
       go e @@ Float.exp (eval env e) *. deriv
    | Pow (e, k) ->
       go e @@ k *. (eval env e) **. (k -. 1.0) *. deriv
  in
  go root 1.0

(** Build a neuron with inputs [xs] and tanh activation. *)
let neuron (lbl : string) (xs : exp list) : exp =
  let ws = List.mapi xs ~f:(fun i _ -> lbl ^ "_w" ^ string_of_int i) in
  let b = lbl ^ "_b" in
  tanh @@ List.fold (List.zip_exn ws xs) ~init:(var b)
            ~f:(fun acc (w, x) -> add acc (mul (var w) x))

(** Build a layer of [nout] neurons with inputs [xs]. *)
let layer (lbl : string) (nout : int) (xs : exp list) : exp list =
  List.init nout ~f:(fun i -> neuron (lbl ^ "_" ^ string_of_int i) xs)

(** Build a multilayer perceptron where the ith layer has [noutsᵢ]
    neurons. Each layer is fed as input to the next, starting with
    [xs] into the first layer. *)
let mlp (nouts : int list) (xs : exp list) : exp list =
  let l = ref xs in
  for i = 0 to List.length nouts - 1 do
    l := layer ("l" ^ string_of_int i) (List.nth_exn nouts i) !l
  done;
  !l

(** Forward evaluation of a list of expressions. *)
let forward (env : float Env.t) (es : exp list) : float list =
  List.map es ~f:(eval env)

let () =

  (* Four input examples. *)
  let xs = [[2.0; 3.0; -1.0];
            [3.0; -1.0; 0.5];
            [0.5; 1.0; 1.0];
            [1.0; 1.0; -1.0]] in
  
  (* And their corresponding labels. *)
  let ys = [1.0; -1.0; -1.0; 1.0] in

  (* Convert the inputs to Num expressions. *)
  let x_exps = List.map xs ~f:(fun l -> List.map l ~f:num) in

  (* Build a forest of MLPs (one for each input example) that produce
     the predicted labels for each input. *)
  let ypreds = List.concat_map x_exps ~f:(mlp [4; 4; 1]) in

  (* Gather the names of all the trainable weights. *)
  let ws = Set.to_list @@ S.union_list @@ List.map ypreds ~f:fvs in

  (* Initialize the weights with random values. *)
  let env = ref (List.fold ws ~init:Env.empty ~f:(fun acc w ->
                     Map.set acc ~key:w ~data:(Random.float_range (-1.0) 1.0))) in

  (* Mean squared error loss expression. *)
  let loss = sum @@ List.map (List.zip_exn ys ypreds) ~f:(fun (ygt, yout) ->
                        pow (sub yout (num ygt)) 2.0) in
  
  (* Run [num_steps] iterations of gradient descent. *)
  let num_steps = 1000 in
  for _ = 0 to num_steps do

    (* Compute gradient of loss wrt. the weights. *)
    let grad = backprop !env loss in

    (* Update each weight by moving it a little bit in the negative
       direction of its partial derivative value. *)
    List.iter (Map.to_alist grad) ~f:(fun (k, v) ->
        env := Map.set !env ~key:k ~data:(Map.find_exn !env k -. 0.1 *. v));

    (* Print current loss value. *)
    print_endline @@ string_of_float @@ eval !env loss
  done;

  (* Print predictions. *)
  print_endline "";
  List.iter (forward !env ypreds) ~f:(fun y ->
      print_endline @@ string_of_float y)
