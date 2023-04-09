use rand::prelude::*;
use sdl2::libc::creat;
use std::time::Duration;
use std::{thread, time::Instant};

mod creature;
use creature::*;

mod world;
use world::*;

extern crate sdl2;
use sdl2::{pixels::Color, render::Canvas, video::Window, EventPump};

const BOX_HEIGHT: i16 = 200;
const BOX_WIDTH: i16 = 200;
const GENOME_SIZE: usize = 16;
const INTERNAL_NEURONS_COUNT: u8 = 4;
const CREATURE_COUNT: usize = 2000;

fn main() {
    /*
    let g = vec![
        0b1000_0001_1000_0000_1111_1111_1111_1111u32,
        0b1000_0100_1000_0110_1111_1111_1111_1111u32,
    ];
    let cr = Creature::from_chromosomes(g, 1);
    let mut wr = World::new(vec![cr], BOX_HEIGHT as usize, BOX_WIDTH as usize);
    loop {
        wr.step();

        println!("{:#?}", wr.creatures[0].position);
    }
    */

    let mut creatures = (0..CREATURE_COUNT * GENOME_SIZE)
        .map(|_| thread_rng().gen::<u32>())
        .collect::<Vec<_>>()
        .chunks(GENOME_SIZE)
        .map(|c| Creature::<GENOME_SIZE>::from_chromosomes(c.try_into().unwrap(), INTERNAL_NEURONS_COUNT))
        .collect::<Vec<_>>();

    let mut world = World::new(&mut creatures, BOX_HEIGHT as usize, BOX_WIDTH as usize);
    let (mut canvas, mut event_pump) = create_canvas().unwrap();
    let mut cn = vec![];

    for generation in 0..20 {
        let zzz = Instant::now();
        event_pump.poll_event();

        for _ in 0..1000 {

            /*
            if generation % 50 == 0 {
                event_pump.poll_event();
                display(&world, &mut canvas);
                thread::sleep(Duration::from_millis(10));
                //world.display();
            }
            */
            world.step(&mut creatures, generation > 2000);
        }
        cn.push(zzz.elapsed().as_micros());
        println!("It took {}µs to do steps", zzz.elapsed().as_micros());
        if generation % 50 == 0 {
            display_survival(&mut canvas);
        }

        let s1 = Instant::now();

        let (killed, filtered, survived) =
            creatures
                .iter()
                .fold((0, 0, vec![]), |(k, f, mut s), c| {
                    if !c.alive {
                        (k + 1, f, s)
                    } else if survive(&c.position) {
                        s.push(c);
                        (k, f, s)
                    } else {
                        (k, f + 1, s)
                    }
                });
        //println!("It took {}µs to do things", s1.elapsed().as_micros());

        let s1 = Instant::now();
        println!(
            "{}: {} survived ({} killed, {} filtered)",
            generation,
            survived.len(),
            killed,
            filtered
        );
        //println!("It took {}µs", s1.elapsed().as_micros());

        let s1 = Instant::now();
        creatures = (0..CREATURE_COUNT)
            .map(|_| {
                (
                    thread_rng().gen::<u32>() as usize % survived.len(),
                    thread_rng().gen::<u32>() as usize % survived.len(),
                )
            })
            .map(|(p1, p2)| survived[p1].reproduce(survived[p2]))
            .collect::<Vec<_>>();
        //println!("It took {}µs to reproduce", s1.elapsed().as_micros());

        //if survived.len() > CREATURE_COUNT * 99 / 100 {
        //    for i in survived.iter().take(10) {
        //        println!("==================================================");
        //        println!("{:#?}", i);
        //        println!("==================================================");
        //        println!("{:#?}", i.neurons);
        //        println!("==================================================");
        //        println!("{:#?}", i.connections);
        //        println!("==================================================");
        //        println!("==================================================");
        //        println!("==================================================");
        //        println!("==================================================");
        //        println!("==================================================");
        //    }
        //}

        let s1 = Instant::now();
        world = World::new(&mut creatures, BOX_HEIGHT as usize, BOX_WIDTH as usize);
        //println!("It took {}µs to create world", s1.elapsed().as_micros());
        //println!("It took {}µs to everything", zzz.elapsed().as_micros());
    }

    let avg: u128 = cn.iter().sum::<u128>() / cn.len() as u128;
    println!("Avg: {avg}µs");
}

fn survive(position: &Position) -> bool {
    //(position.x < 50 || position.x > BOX_WIDTH - 50) && position.y < 50
    //(position.x < 70 || position.x > BOX_WIDTH - 70) && (position.y < 70 || position.y > BOX_HEIGHT - 70)

    /*(position.x - BOX_WIDTH / 2).pow(2) +
    (position.y - BOX_HEIGHT / 2).pow(2) < 1200*/

    position.x < 22
}

fn create_canvas() -> Option<(Canvas<Window>, EventPump)> {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = match sdl_context.video() {
        Ok(s) => s,
        Err(s) => {
            println!("{}", s);
            return None;
        }
    };
    let window = match video_subsystem
        .window(
            "rust-sdl2 demo",
            BOX_HEIGHT as u32 * 4,
            BOX_WIDTH as u32 * 4,
        )
        .position_centered()
        .build()
    {
        Ok(s) => s,
        Err(s) => {
            println!("{}", s);
            return None;
        }
    };

    let sdl_event_pump = sdl_context.event_pump().unwrap();

    //window.set_bordered(false);
    //window.set_fullscreen(FullscreenType::True).unwrap();
    let mut canvas = match window.into_canvas().build() {
        Ok(s) => s,
        Err(s) => {
            println!("{}", s);
            return None;
        }
    };
    canvas.set_scale(4f32, 4f32).unwrap();

    Some((canvas, sdl_event_pump))
}

fn display(world: &World<GENOME_SIZE>, creatures: &[Creature<GENOME_SIZE>], canvas: &mut Canvas<Window>) {
    canvas.set_draw_color(Color::BLACK);
    canvas.clear();
    canvas.set_draw_color(Color::WHITE);

    let mut killers: Vec<sdl2::rect::Point> = vec![];
    let mut others: Vec<sdl2::rect::Point> = vec![];

    for c in creatures.iter().filter(|c| c.alive) {
        if c.neurons
            .iter()
            .any(|n| matches!(n, Neuron::Action(ActionType::Kill)))
        {
            killers.push((&c.position).into());
        } else {
            others.push((&c.position).into());
        }
    }

    canvas.set_draw_color(Color::WHITE);
    canvas.draw_points(&others[..]).unwrap();

    canvas.set_draw_color(Color::RED);
    canvas.draw_points(&killers[..]).unwrap();

    canvas.present();
}

fn display_survival(canvas: &mut Canvas<Window>) {
    canvas.set_draw_color(Color::RED);
    for x in 0..BOX_WIDTH {
        for y in 0..BOX_HEIGHT {
            let pos = Position { x, y };
            if survive(&pos) {
                canvas.draw_point(&pos).unwrap();
            }
        }
    }

    canvas.present();
}

impl Into<sdl2::rect::Point> for &Position {
    fn into(self) -> sdl2::rect::Point {
        sdl2::rect::Point::new(self.x as i32, self.y as i32)
    }
}

/*
impl ActionType {
    pub fn to_action_result(&self, creature: &Creature) -> ActionResult {
        match self {
            ActionType::Move(d) => ActionResult::Move(creature.position + *d),
            ActionType::MoveForward => ActionResult::Move(creature.position + creature.last_move),
            ActionType::Kill => ActionResult::Kill(creature.position + creature.last_move),
        }
    }
}
*/
